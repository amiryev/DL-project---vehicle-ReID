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

You can test the Text2Seg on demo.ipynb notebook. 


## Citing Text2Seg

If you find Text2Seg useful, please use the following BibTeX entry.

```
@article{zhang2023text2seg,
  title={Text2Seg: Remote Sensing Image Semantic Segmentation via Text-Guided Visual Foundation Models},
  author={Zhang, Jielu and Zhou, Zhongliang and Mai, Gengchen and Mu, Lan and Hu, Mengxuan and Li, Sheng},
  journal={arXiv preprint arXiv:2304.10597},
  year={2023}
}
```
