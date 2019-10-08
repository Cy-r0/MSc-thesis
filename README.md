# One-Stage Instance Segmentation with a Boundary Distance Representation

## Description
One-stage deep learning pipeline for instance segmentation in images. Model takes in an image and generates a semantic segmentation and a watershed representation. These two are combined in postprocessing.

## Requirements
Repo works on Ubuntu 18.04.2 LTS with:
- pytorch 1.1.0
- torchvision 0.3.0
- scikit-learn 0.21.2
- numpy 1.16.4
- pandas 0.24.2
- pillow 6.0.0
- opencv 3.1.0
- matplotlib 3.1.0
- seaborn 0.9.0
- tensorboardx 1.7
Additionally, PASCAL VOC 2012 devkit with augmented set needs to be downloaded and its root referenced in config.py.

## Installing
1. Install dependencies
2. Clone repo
3. Download PASCAL VOC 2012 devkit from http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit
4. Download augmented PASCAL set from https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0 and put it in the PASCAL devkit, in the Segmentation task.
