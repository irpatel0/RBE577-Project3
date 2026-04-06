# RL Agent Setup

This project trains and evaluates a policy for `LunarLander-v2` using PyTorch.

## 1. Create the Conda environment

```bash
conda create -n rl-agent python=3.8 -y
conda activate rl-agent
```

Install a pip version that is compatible with `gym==0.21.0`:

```bash
python -m pip install "pip==23.0.1" "setuptools==65.5.0" "wheel==0.38.4"
```

## 2. Install dependencies

Install the core scientific Python packages and `ffmpeg` for video export:

```bash
conda install -y numpy matplotlib ipython ffmpeg
```

Install direct video-writing support used by `run_lunar_lander`:

```bash
pip install "imageio[ffmpeg]"
```

Install the CUDA-enabled PyTorch build:

```bash
pip install torch==2.4.1+cu121 torchvision==0.19.1+cu121 torchaudio==2.4.1+cu121 \
	--index-url https://download.pytorch.org/whl/cu121
```

This installs the GPU build for CUDA 12.1. It requires a working NVIDIA driver on the machine.

Install the reinforcement learning environment packages used by this repo:

```bash
pip install gym==0.21.0 Box2D pygame "pyglet<2"

```

Install PyBullet for the KUKA Env in PyBullet:
```bash
pip install pybullet==3.2.6

```

Install some helpers:
```bash

pip install pyyaml

```


To confirm PyTorch can see the GPU, run:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device count:', torch.cuda.device_count()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
```

## 4. Run training

From the repository root:

```bash
python train.py
```

The training script will:

- create a random-policy video attempt,
- train the actor-critic agent,
- save a reward plot under `plots/`,
- save a checkpoint under `checkpoints/`,
- create trained-policy video attempts.

## 5. Run evaluation

To evaluate a saved checkpoint:

```bash
python eval.py --checkpoint checkpoints/lunar_lander_actor.pt --episodes 5
```

To also save an evaluation video:

```bash
python eval.py --checkpoint checkpoints/lunar_lander_actor.pt --episodes 5 --video eval_policy.mp4
```

## 6. Configuration

Training and evaluation settings live in `config.json`, including:

- environment id,
- algorithm selection,
- learning rates,
- rollout length,
- checkpoint path.

Edit `config.json` before running `train.py` if you want different hyperparameters.