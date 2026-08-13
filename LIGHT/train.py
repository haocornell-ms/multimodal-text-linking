import os
import sys
import random
import numpy as np
import yaml
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import LinkingTrainDataset, LinkingTestDataset
from dataset.buildin import DATASET_META
from models.text_linking import LightTextLinking
from models.model_utils import get_processors
from utils.options import parse_args
from utils.utils import *


def count_parameters(model, trainable_only=True):
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def build_optimizer(model, args):
    """Use conservative LRs for pretrained modules and a larger visual LR."""
    visual_prefixes = (
        "word_style_encoder.",
        "visual_edge_scorer.",
        "visual_edge_scale_logit",
    )
    backbone_parameters = [
        parameter for parameter in model.light.parameters() if parameter.requires_grad
    ]
    backbone_parameter_ids = {id(parameter) for parameter in backbone_parameters}
    visual_parameters = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and name.startswith(visual_prefixes)
    ]
    visual_parameter_ids = {id(parameter) for parameter in visual_parameters}
    task_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
        and id(parameter) not in backbone_parameter_ids
        and id(parameter) not in visual_parameter_ids
    ]

    backbone_lr = getattr(args, "backbone_lr", args.lr)
    base_task_lr = getattr(args, "base_task_lr", backbone_lr)
    parameter_groups = [
        {"params": backbone_parameters, "lr": backbone_lr, "name": "light_backbone"},
        {"params": task_parameters, "lr": base_task_lr, "name": "base_linking_heads"},
        {"params": visual_parameters, "lr": args.lr, "name": "visual_residual"},
    ]
    parameter_groups = [group for group in parameter_groups if group["params"]]
    optimizer = AdamW(
        parameter_groups,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )
    for group in optimizer.param_groups:
        num_parameters = sum(parameter.numel() for parameter in group["params"])
        print(
            f"Optimizer group {group['name']}: {num_parameters:,} parameters, "
            f"lr={group['lr']:.2e}"
        )
    return optimizer


def set_pretrained_base_trainable(model, trainable):
    """Freeze/unfreeze everything except the newly introduced visual branch."""
    visual_prefixes = (
        "word_style_encoder.",
        "visual_edge_scorer.",
        "visual_edge_scale_logit",
    )
    for name, parameter in model.named_parameters():
        if not name.startswith(visual_prefixes):
            parameter.requires_grad = trainable


def validate(model, data_loader, device, log_writer, epoch=None):
    model.eval()
    total_losses = defaultdict(int)
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            input_data = {k: v.to(device) for k, v in batch.items()}
            _, losses = model(input_data, return_loss=True)
            for k, loss in losses.items():
                total_losses[k] += loss.item() / len(data_loader)
                
    if model.use_factorized_linking:
        val_loss = sum(total_losses.values())
    else:
        # Preserve the original checkpoint-selection criterion for the joint
        # successor model; auxiliary losses shape training but do not select it.
        val_loss = total_losses['base_loss']
    for k, loss in total_losses.items():
        print(f"Epoch {epoch + 1}: Validation {k}: {loss}")
        log_writer.add_scalar(f"Loss/Val_{k}", loss, epoch)
    print(f"Epoch {epoch + 1}: Validation selection_loss: {val_loss}")
    return val_loss


def main():
    seed_everything(1234)
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'GPU Device: {device}')
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    log_writer = SummaryWriter(log_dir=args.log_dir)
    with open(args.save_config_file, 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)
    print(f"Configuration saved to {args.save_config_file}")

    ###
    ### dataloader
    tokenizer, image_processor = get_processors(args.pretrained_model_name)
    train_dataset = LinkingTrainDataset(tokenizer, image_processor, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = LinkingTestDataset(tokenizer, image_processor, args, mode='val', return_ori=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)    

    ###
    ### model
    model = LightTextLinking(args)        
    full_pretrained_weights = getattr(args, "full_pretrained_weights", None)
    if full_pretrained_weights is not None:
        assert os.path.exists(full_pretrained_weights), "Full pretrained weights must exist"
        print("... Loading complete fine-tuned text/linking baseline ...")
        checkpoint = load_model_weights(full_pretrained_weights)
        if checkpoint and next(iter(checkpoint)).startswith('module.'):
            checkpoint = {key[len('module.'):]: value for key, value in checkpoint.items()}
        msg = model.load_state_dict(checkpoint, strict=False)
        print(msg)
    elif args.light_pretrained_weights is not None:
        assert os.path.exists(args.light_pretrained_weights), "LIGHT pretrained weights must exists"
        print("... Loading pretrained weights for LIGHT ...")
        checkpoint = load_model_weights(args.light_pretrained_weights)
        checkpoint = {k[len("module.light."):]: v for k, v in checkpoint.items() if k.startswith('module.light.')}    
        msg = model.light.load_state_dict(checkpoint, strict=False)
        print(msg) 

    total_params = count_parameters(model, trainable_only=False)
    print(f"Total parameters: {total_params:,}")

    optimizer = build_optimizer(model, args)
    freeze_base_epochs = getattr(args, "freeze_base_epochs", 0)
    if freeze_base_epochs > 0:
        set_pretrained_base_trainable(model, False)
        print(f"Freezing pretrained baseline for {freeze_base_epochs} epochs")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        patience=args.scheduler_patience, 
        factor=0.1, 
        threshold=0,
        min_lr=[0.0000005, 0.0000005, 0.000005],
        verbose=True)
    
    model.to(device)

    ###
    ### training
    best_val_loss, epochs_no_improve = np.inf, 0
    min_epochs_before_early_stop = getattr(args, "min_epochs_before_early_stop", 0)
    if min_epochs_before_early_stop > 0:
        print(
            "Early stopping disabled for the first "
            f"{min_epochs_before_early_stop} epochs"
        )
    
    for epoch in range(args.num_epochs):
        if epoch == freeze_base_epochs and freeze_base_epochs > 0:
            set_pretrained_base_trainable(model, True)
            print(f"Epoch {epoch + 1}: Unfroze pretrained baseline parameters")
        model.train()
        train_dataset.reset()
        
        epoch_losses = defaultdict(int)
        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{args.num_epochs}"):
            input_data = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            _, losses = model(input_data, return_loss=True)

            total_loss = 0
            for k, loss in losses.items():
                epoch_losses[k] += loss.item() / len(train_dataloader)
                total_loss += loss
                 
            total_loss.backward()
            optimizer.step()

        for k, loss in epoch_losses.items():
            print(f"Epoch {epoch + 1}: Training {k}: {loss}")
            log_writer.add_scalar(f"Loss/Train_{k}", loss, epoch)
        if model.visual_edge_scale is not None:
            visual_edge_scale = model.visual_edge_scale.detach().item()
            print(f"Epoch {epoch + 1}: Visual edge residual scale: {visual_edge_scale:.6f}")
            log_writer.add_scalar("Model/Visual_edge_scale", visual_edge_scale, epoch)

        ### save model
        if args.save_every_epoch > 0 and (epoch + 1) % args.save_every_epoch == 0:
            model_save_path = os.path.join(args.checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at epoch {epoch + 1}: {model_save_path}")

        ### validate
        if (epoch + 1) % args.eval_every_epoch == 0:
            val_loss = validate(model, val_dataloader, device, log_writer=log_writer, epoch=epoch)
            if val_loss < best_val_loss:
                print(f"Loss decreases: {best_val_loss-val_loss}, Best model saved.")            
                model_save_path = os.path.join(args.output_dir, "best_model.pth")
                torch.save(model.state_dict(), model_save_path)
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"No improvement in val loss for {epochs_no_improve} epochs.")
                    
            scheduler.step(val_loss)

        ### terminate
        completed_epochs = epoch + 1
        patience_exhausted = (
            completed_epochs >= min_epochs_before_early_stop
            and epochs_no_improve >= args.patience
        )
        if patience_exhausted or epoch == args.num_epochs - 1:
            base_comm = ("python inference.py --test_dataset MapText_test "
                         "--out_file predict.json --model_dir {model_dir} "
                         "--anno_path {anno_path} --img_dir {img_dir}")
            if args.val_dataset == "MapText_val":
                comm = base_comm.format(model_dir=args.output_dir, 
                                        anno_path=DATASET_META['MapText_test']['anno_path'], 
                                        img_dir=DATASET_META['MapText_test']['img_dir'])
                print(comm)
                os.system(comm)
            if args.val_dataset == "HierText_val":
                comm = base_comm.format(model_dir=args.output_dir, 
                                        anno_path=DATASET_META['HierText_test']['anno_path'], 
                                        img_dir=DATASET_META['HierText_test']['img_dir'])
                print(comm)
                os.system(comm)
            sys.exit(0)
            
        
    
if __name__ == '__main__':
    main()
