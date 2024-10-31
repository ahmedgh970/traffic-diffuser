## The official PyTorch implementation of TrafficDiffuser <br><sub>Denoising Diffusion Model for Traffic Simulation</sub>
This project features a traffic simulation model for conditional trajectory generation using diffusion models. The model is capable of conditioning on agent histories and environmental maps, generating plausible trajectories for multiple agents with variable sequence length.

![TrafficDiffuser overall architecture](docs/asset/TrafficDiffuser.png)

The repository is organized as follows:
  * [Documentation](#documentation)
  * [Folder Structure](#folder-structure)
  * [Setup](#setup)
  * [Data Processing](#data-processing)
  * [Training](#training)
  * [Sampling and Evaluation](#sampling-and-evaluation)
  * [Backbone Architecture](#backbone-architecture)

## Overview
TrafficDiffuser is a PyTorch-based implementation of a conditional trajectory generation model for traffic simulation. It leverages denoising diffusion models to simulate realistic traffic scenarios. 

## Documentation
* Refer to [ScenarioNet](https://github.com/metadriverse/scenarionet) to convert Nuscenes, Waymo, nuPlan, and Argoverse datasets into a unified dict format, which is required before training and evaluating TrafficDiffuser.
* The diffusion process is modified from OpenAI's diffusion repos: [GLIDE](https://github.com/openai/glide-text2im/blob/main/glide_text2im/gaussian_diffusion.py), [ADM](https://github.com/openai/guided-diffusion/blob/main/guided_diffusion), and [IDDPM](https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/gaussian_diffusion.py).
* Refer to [Vim](https://github.com/hustvl/Vim), as the official implementation of the paper [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417).

## Folder Structure
``` 
traffic-diffuser-main
├── diffusion/                    # Diffusion dir
├── docs/asset                    # Documentation figures               
├── models/                       # Backbones dir
│   └── layers.py                 # Layers and utility functions
│   └── model_td.py               # TrafficDiffuser backbone
├── requirements.txt              # Requirements
├── process_dataset.ipynb         # Data processing pipeline
├── metrics.py                    # Evaluation metrics               
├── sample.py                     # Sampling script 
└── train.py                      # Training script
```

## Setup

First, download and set up the repo:

```bash
git clone https://github.com/gen-TII/traffic-diffuser.git
cd traffic-diffuser
```

Then, create a python 3.10 conda env and install the requirements

```bash
# Install TrafficDiffuser
conda create --name venv python=3.10
conda activate venv
pip install -r requirements.txt

# Install MetaDrive Simulator
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e.

# Install ScenarioNet
git clone https://github.com/metadriverse/scenarionet.git
cd scenarionet
pip install -e .
```

## Data Processing
First, convert and merge the original datasets (nuscenes, waymo, etc.) into unified dictionary-formatted pickle files using ScenarioNet. Next, preprocess these pickle files to generate the data and map directories in the required format. Lastly, complete the data processing pipeline to obtain the final rasterized maps, along with the final training and testing scenarios. Follow the [`preprocess_dataset.ipynb`](process_dataset.ipynb) notebook to perform these three data processing steps on nuScenes and Waymo.


## Training
We provide a training script for TrafficDiffuser model in [`train.py`](train.py).

To launch TrafficDiffuser-L training with `N` GPUs on one node:
```bash
accelerate launch train.py --model-module model_td --model TrafficDiffuser-L \
  --max-num-agents 46 --hist-length 8 --seq-length 5 --use-map-embed --use-ckpt-wrapper \
    --train-dir /path/to/preprocessed/scenarios/ --map-train-dir /path/to/maps/
```

To launch TrafficDiffuser-L training with `1` GPU (id=1):
```bash
accelerate launch --num-processes=1 --gpu_ids 1 train.py --model-module model_td --model TrafficDiffuser-L \
  --max-num-agents 46 --hist-length 8 --seq-length 5 --use-map-embed --use-ckpt-wrapper \
    --train-dir /path/to/preprocessed/scenarios/ --map-train-dir /path/to/maps/
```


## Sampling and Evaluation

To sample trajectories from a pretrained TrafficDiffuser-L model, using GPU id 1, run:
```bash
python sample.py --cuda-device 1 --model-module model_td --model TrafficDiffuser-L \
  --max-num-agents 46 --hist-length 8 --seq-length 5 --num-sampling-steps 250 --num-sampling 100 --use-map-embed \
    --ckpt /path/to/model.pt --test-dir /path/to/preprocessed/test/scenarios/ --map-test-dir /path/to/test/maps/
```

The sampling results are automatically saved in the model's designated results directory, organized within the samples subfolder for easy access. Additionally, evaluation metrics such as FD (Fréchet Distance), ADE (Average Displacement Error), and others are logged for each test scenarios. The evaluation log file alse include the model summary, number of parameters, FLOPs, and inference runtime.

An example of the evaluation log file results:
```bash
...
The average evaluation results across all scenarios:
 - Frechet Distance (FD): 2.3737549884231917
 - Absolute Traveled Distance Difference (ATDD): 1.7922132715630574
 - Polygone Area (PA): 3.3179398003974447
 - Average Displacement Error (ADE): 1.2074199433614015
 - Final Displacement Error (FDE): 2.364478100013666
```

#### Visualization of test scenarios 4, 7 and 18:
![TrafficDiffuser-L sampling results](docs/asset/Visualizations.png)


## Backbone Architecture
![TrafficDiffuser diffusion backbone](docs/asset/Backbone.png)