# Object Reidentification in remote sensing

This is a project in Deep Learning 0510-7255-02 course 

## Atuhers
Amir Yevnin

Dor Sivan

Adi ???

## Pre-requisits

We assume you already have basic packages like numpy and torch installed.
In order for this project to work you must install the following model: GroundingDINO, CLIP, Segment Anything 2:

1. Install GroundingDIno:

```
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip3 install -q -e .
cd ..
mkdir Pretrained
cd Pretrained
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..
```

2. Download CLIP Surgery repository
```
git clone https://github.com/xmed-lab/CLIP_Surgery.git
```

3. Install CLIP repository
```
pip install git+https://github.com/openai/CLIP.git
```

4. Install Segment Anything 2:

```
pip install opencv-python matplotlib
pip install 'git+https://github.com/facebookresearch/segment-anything-2.git'

mkdir -p checkpoints/
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt
wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt
```



## <a name="GettingStarted"></a>Getting Started

There are 2 options to run this algorithm:

1. Run over dataset
```
For this option use main.py.
You will have to provide the dataset in the /data directory.
```
3. Run app on two images
```
For this option use app.py.
You will have to provide a query image and a base image.
```
