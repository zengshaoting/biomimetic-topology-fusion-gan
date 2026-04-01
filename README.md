# Biomimetic Topology Fusion GAN

Source code accompanying the manuscript:

**Biomimetic Topology Fusion Generation Using Generative Adversarial Networks: Design and Evaluation Balancing Mechanical Performance and Aesthetic Quality**

## Overview
This repository contains a PyTorch implementation of a CycleGAN-based workflow for biomimetic topology fusion generation. The project is designed to learn unpaired image-to-image mappings between two morphology domains and generate fused structures that balance **mechanical performance** and **aesthetic quality**.

In the manuscript, the two domains correspond to:
- **Performance-oriented morphologies** (for example, dragonfly wing venation and leaf venation)
- **Aesthetics-oriented morphologies** (for example, honeycomb cells and pinecone spiral patterns)

The codebase supports:
- unpaired dataset loading for domain A and domain B;
- CycleGAN model training;
- checkpoint saving and loss logging;
- inference / sample generation from trained checkpoints;
- saving generated results for downstream evaluation and visualization.

## Repository structure

```text
.
├── CycleGAN.ipynb                # notebook version / interactive experimentation
├── train.py                      # training entry point
├── test.py                       # inference / sample generation entry point
├── requirements.txt              # minimal dependency list provided with the project
├── datasetTools/
│   ├── __init__.py               # dataset factory / dataloader wrapper
│   ├── base_dataset.py           # shared dataset utilities and transforms
│   ├── image_folder.py           # image file discovery helpers
│   └── unaligned_dataset.py      # unpaired dataset loader for trainA/trainB/testA/testB
├── models/
│   ├── __init__.py               # model factory
│   ├── base_model.py             # shared training / checkpoint logic
│   ├── cyclegan_model.py         # CycleGAN implementation
│   └── networks.py               # generator / discriminator definitions
├── options/
│   ├── base_options.py           # common CLI arguments
│   ├── train_options.py          # training-specific CLI arguments
│   └── test_options.py           # test-specific CLI arguments
├── utilities/
│   ├── image_pool.py             # image replay buffer
│   ├── renderer.py               # logging / image saving utilities
│   ├── samples.py                # sample output helper
│   ├── util.py                   # HTML helper utilities
│   └── utilities.py              # general image / file helpers
├── checkpoints/                  # saved model checkpoints (generated at runtime)
└── results/                      # generated results (generated at runtime)
```

## What this code does
The repository implements a standard CycleGAN training pipeline tailored for morphology fusion tasks:

1. **Load unpaired images** from two domains using the `unaligned_dataset` loader.
2. **Train two generators and two discriminators** to perform bidirectional translation between the two morphology domains.
3. **Apply adversarial and cycle-consistency constraints** to preserve structural content while learning cross-domain style / morphology transfer.
4. **Save checkpoints and intermediate samples** during training.
5. **Generate translated / fused morphology images** during testing for downstream visualization, aesthetic evaluation, and later 3D modeling.

## Dataset format
The code expects an unpaired dataset directory with separate folders for domain A and domain B.

```text
/path/to/dataset/
├── trainA/
├── trainB/
├── testA/
└── testB/
```

Where:
- `trainA/` contains training images from domain A
- `trainB/` contains training images from domain B
- `testA/` contains test images from domain A
- `testB/` contains test images from domain B

For the manuscript use case, you may define:
- **Domain A** = aesthetics-oriented natural morphology images
- **Domain B** = performance-oriented natural morphology images

or reverse this assignment depending on the translation direction you want to emphasize.

## Environment and dependencies
The included `requirements.txt` lists the following packages:
- `torch>=0.4.1`
- `torchvision>=0.2.1`
- `dominate>=2.3.1`
- `visdom>=0.1.8.3`

Based on the source files in this repository, you should also ensure that the following packages are available in your environment:
- `numpy`
- `Pillow`

A simple setup flow is:

```bash
pip install -r requirements.txt
pip install numpy pillow
```

If you are using a modern PyTorch environment, it is also fine to create a fresh conda or venv environment and install current compatible versions of these packages.

## Recommended environment
Example using conda:

```bash
conda create -n biomimetic-cyclegan python=3.10
conda activate biomimetic-cyclegan
pip install -r requirements.txt
pip install numpy pillow
```

## Training
The main training script is `train.py`.

### Minimal training command

```bash
python train.py \
  --dataset_path ./datasets/your_dataset \
  --name biomimetic_topology_fusion \
  --model cyclegan
```

### Example training command for biomimetic topology fusion

```bash
python train.py \
  --dataset_path ./datasets/biomimetic_morphologies \
  --name dragonfly_honeycomb_cyclegan \
  --model cyclegan \
  --direction AtoB \
  --batch_size 1 \
  --load_size 256 \
  --preprocess resize_and_crop \
  --niter 100 \
  --niter_decay 100 \
  --lr 0.0002 \
  --beta1 0.5 \
  --lambda_A 10.0 \
  --lambda_B 10.0 \
  --lambda_identity 0.5
```

### Key training arguments
- `--dataset_path`: path to the dataset root containing `trainA`, `trainB`, `testA`, and `testB`
- `--name`: experiment name; used to create the checkpoint directory
- `--model`: model type; use `cyclegan`
- `--direction`: translation direction (`AtoB` or `BtoA`)
- `--batch_size`: batch size (default is 1)
- `--load_size`: resize dimension before conversion to tensor
- `--niter`: number of epochs at the initial learning rate
- `--niter_decay`: number of epochs for linearly decaying the learning rate
- `--lambda_A`, `--lambda_B`: cycle-consistency weights
- `--lambda_identity`: identity mapping weight
- `--gpu_ids`: GPU selection, e.g. `0` or `-1` for CPU

## Training outputs
During training, the code creates outputs under:

```text
./checkpoints/<experiment_name>/
```

Typical outputs include:
- saved network checkpoints;
- `loss_log.txt` with printed training losses;
- `samples/` directory with saved images such as `epochXXX_real_A.png`, `epochXXX_fake_B.png`, etc.

## Inference / testing
The main inference script is `test.py`.

### Important note
`options/test_options.py` sets the default model to `test`, but this repository only includes `cyclegan_model.py`. Therefore, when running inference, you should explicitly pass:

```bash
--model cyclegan
```

### Minimal test command

```bash
python test.py \
  --dataset_path ./datasets/your_dataset \
  --name biomimetic_topology_fusion \
  --model cyclegan \
  --load_epoch latest
```

### Example test command

```bash
python test.py \
  --dataset_path ./datasets/biomimetic_morphologies \
  --name dragonfly_honeycomb_cyclegan \
  --model cyclegan \
  --phase test \
  --load_epoch latest \
  --num_test 50
```

Generated images are saved under:

```text
./results/<experiment_name>/<phase>_<epoch>/
```

## Typical workflow for this project
1. Organize biomimetic images into `trainA`, `trainB`, `testA`, and `testB`.
2. Train a CycleGAN model for a selected morphology pair.
3. Inspect saved checkpoints and generated sample images during training.
4. Run `test.py` on the trained model to export generated fusion morphologies.
5. Use generated outputs for later quantitative evaluation, visualization, or 3D reconstruction steps described in the manuscript.

## Notes on domain assignment
You can define the two domains according to your experimental design. For example:

- **A = aesthetics-oriented**, **B = performance-oriented**
- or **A = performance-oriented**, **B = aesthetics-oriented**

The `--direction` flag controls which domain is treated as input and which as target during loading and translation.

## Reproducibility notes
To improve reproducibility, consider documenting the following together with each experiment:
- exact dataset composition;
- image preprocessing settings;
- training command;
- checkpoint epoch used for testing;
- hardware / GPU information;
- any manual post-processing applied after image generation.

## Known limitations / implementation notes
- This repository currently provides the core CycleGAN training and testing pipeline, but does **not** include a full end-to-end pipeline for the later stages described in the manuscript (for example, 3D modeling, mechanical testing, or quantitative aesthetic analysis scripts), unless those files are added separately.
- The supplied `requirements.txt` is minimal and may need supplementation depending on your environment.
- The default test option should be overridden with `--model cyclegan` when running `test.py`.
- The repository currently does not include a license file; add one before public release if you plan to archive the code.

## Suggested citation
If you release this code alongside the manuscript, you may cite it as:

```text
Zeng, S., Shi, Y., Zhang, Z., Yu, Y. & Liu, Y. Source code for: Biomimetic Topology Fusion Generation Using Generative Adversarial Networks: Design and Evaluation Balancing Mechanical Performance and Aesthetic Quality (Version 1.0.0). Zenodo. https://doi.org/[DOI]
```

## Code availability statement (draft)
The manuscript-associated source code used for CycleGAN training and inference for biomimetic topology fusion generation has been archived in Zenodo and is publicly available at https://doi.org/[DOI]. The archived version corresponding to this submission is version 1.0.0. The source code is additionally maintained on GitHub at [GitHub repository URL] for version control, while the Zenodo record is the citable version of record.

## Authors
- Shaoting Zeng
- Yifeng Shi
- Zeyi Zhang
- Yujuan Yu
- Yuxin Liu

Affiliation: College of Art and Design, Beijing University of Technology, Beijing 100124, China

Corresponding author: **Shaoting Zeng**

## Contact
- Shaoting Zeng — `sjmjzst@gmail.com`

## Acknowledgement
This work is associated with the manuscript *Biomimetic Topology Fusion Generation Using Generative Adversarial Networks: Design and Evaluation Balancing Mechanical Performance and Aesthetic Quality*.
