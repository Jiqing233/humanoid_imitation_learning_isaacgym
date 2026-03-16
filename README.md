## Environment Setup

### 1. Create a Conda Environment

Create and activate a Python 3.8 conda environment:

```bash
conda create -n isaac_gym python=3.8
conda activate isaac_gym
```

### 2. Install PyTorch 2.0

Install PyTorch with CUDA 11.7 support:

```bash
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia
```

### 3. Install Isaac Gym Preview 4
Go to the NVIDIA developer website:
[Isaac Gym Download Page](https://developer.nvidia.com/isaac-gym)

Follow and Download: IsaacGym_Preview_4_Package.tar.gz, unzip, cd to python/ directory, install by:

```bash
pip install -e .
```
Test installation is complete:

```bash
cd examples/
python joint_monkey.py
```

