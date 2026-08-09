# Training the word-style and relation-aware LIGHT model

This variant preserves LIGHT's LayoutLMv3 and polygon encoder and adds:

1. A small CNN over original-resolution word crops.
2. Explicit appearance, scale, and orientation descriptors.
3. A directional pairwise relation bias and supervised style contrastive loss.

Both new output layers are zero-initialized, so a pretrained LIGHT checkpoint starts
with the same linking scores as the original head and learns the new residual signals.

## Recommended workflow

Use the local CPU machine for code checks and inspecting samples, then use a persistent
Runpod volume for training. Do not train the full model locally: LayoutLMv3 plus 1,000
tokens is impractically slow on CPU. Keep the repository, datasets, Hugging Face cache,
and checkpoints on the persistent volume so a stopped pod does not lose them.

For fastest iteration, first copy only a small data subset to Runpod and verify one
epoch end to end. Start the long job only after inference and evaluation work on that
checkpoint.

## Environment on Runpod

The repository environment pins CUDA 12.1 builds of PyTorch. Choose a CUDA 12.1 image
with Python 3.10, then from the repository root run:

```bash
conda env create -f env.yaml
conda activate map
cd LIGHT
```

If the image does not contain Conda, create a Python 3.10 virtual environment and
install the packages under `pip:` in `env.yaml`. Confirm the GPU before training:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Set dataset paths in `dataset/buildin.py`. Keep images on local/persistent NVMe rather
than an object-store mount because every sample reads the original map and extracts
word crops.

## CPU and one-batch checks

On either machine:

```bash
cd LIGHT
python -m py_compile dataset/dataset.py models/text_linking.py train.py inference.py
```

On Runpod, run a one-sample inspection before training. Verify that `word_crops` has
shape `[256, 3, 32, 128]`, `word_styles` is `[256, 15]`, and `word_geometries` is
`[256, 7]`. Display several non-padded crops if possible; crop errors are much cheaper
to find here than after training.

## Stage 0: reproduce the baseline

Make a copy of `configs/light.yaml` and set:

```yaml
use_word_style: False
use_pairwise_relations: False
aux_losses: ["bidirection", "focal"]
exp_name: "light_baseline_repro"
version: 1
```

Train and evaluate this once. It establishes whether the new environment, data paths,
and supplied pretrained weight reproduce the expected baseline.

```bash
python train.py --config configs/light_baseline_repro.yaml
```

## Stage 1: pairwise relations only

Use:

```yaml
use_word_style: False
use_pairwise_relations: True
aux_losses: ["bidirection", "focal"]
exp_name: "light_relations"
version: 1
lr: 0.00005
```

This isolates gains from normalized displacement, scale ratios, direction, orientation,
and alignment. It adds very few parameters and should be the first ablation.

## Stage 2: style fusion without contrastive loss

Use both branches but initially omit the auxiliary style objective:

```yaml
use_word_style: True
use_pairwise_relations: True
aux_losses: ["bidirection", "focal"]
style_contrastive_weight: 0.0
exp_name: "light_style_relations"
version: 1
```

This tells whether successor supervision alone learns useful crop/style features.

## Stage 3: full model

Use the checked-in `configs/light.yaml`, whose relevant settings are:

```yaml
use_word_style: True
use_pairwise_relations: True
aux_losses: ["bidirection", "focal", "style_contrastive"]
style_contrastive_weight: 0.1
style_temperature: 0.1
```

Run:

```bash
python train.py --config configs/light.yaml
```

The total training loss already sums all returned losses. The configured contrastive
weight is applied inside the model, so do not multiply it again in `train.py`.

## GPU choice and memory

An RTX 4090, A5000/A6000, L40S, or A100 is appropriate. Begin with the existing batch
size of 2. The padded crops add about 12 MiB per sample (24 MiB for batch size 2) before activations. If memory is
tight, use this order:

1. Reduce `max_style_words` from 256 to 128 only if maps rarely exceed 128 retained words.
2. Reduce batch size from 2 to 1.
3. Reduce crop width from 128 to 96.

Do not reduce global token length until measuring how many annotations are truncated.
Any words above `max_style_words` still use the original LIGHT embedding and pair
scoring, but receive zero style/geometry additions; set the cap high enough to cover at
least the 95th percentile of words per sample.

## Hyperparameter search

Keep the first search small:

| Parameter | Values |
|---|---|
| learning rate | `2e-5`, `5e-5` |
| style contrastive weight | `0.03`, `0.1`, `0.3` |
| crop context | `0.15`, `0.25`, `0.4` |
| crop width | `96`, `128` |

Select by validation edge F1 or the official linking metric, not training loss. The
contrastive term changes loss scale, so validation loss is not directly comparable
between experiments with different weights.

Run at least three seeds for the final baseline and best configuration. The dataset's
reported training length is only 200 iterations for MapText, so single-run variance can
be material.

## Monitoring

Start TensorBoard from the repository's persistent volume:

```bash
tensorboard --logdir LIGHT/_runs --bind_all --port 6006
```

Watch the base, bidirectional, focal, and style-contrastive losses separately. Stop and
inspect the inputs if the style loss is NaN, if it remains exactly zero on almost every
batch, or if validation base loss degrades immediately. Zero style loss can legitimately
occur for a batch containing no group with at least two retained words.

After training, run inference using the experiment directory containing `config.yaml`
and `best_model.pth`:

```bash
python inference.py \
  --test_dataset MapText_test \
  --out_file predict.json \
  --prob_out_file top3_probabilities.json \
  --model_dir _runs/EXP_NAME__vVERSION \
  --anno_path /path/to/test.json \
  --img_dir /path/to/test/images
```

Compare edge precision, recall, F1, and complete-group accuracy for every stage. Also
stratify by word height, background contrast, orientation difference, and inter-word
distance; these slices reveal whether the new branch is improving the intended cases.

## Checkpoint compatibility

`train.py` loads only the pretrained `model.light` weights and initializes the new
linking modules normally. Inference loads full fine-tuned checkpoints with
`strict=False`, so older checkpoints remain readable when the new config fields are
present. For a fair ablation, always start each stage from the same LIGHT pretrained
checkpoint rather than continuing from a different ablation's best model.
