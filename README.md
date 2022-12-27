# Context

This code is part of a project to detect anomalies in pipes. For this a robot directed by an operator takes a video, this code aims to classify frames extracted from these videos between presence of crack or absence.

# Installation

Just clone the repo, possibly add frames annotated in data folder.

# Minimal operation

In the file train.py (line 108), select the model that we want to run.  
Execute train.py 

# Description

custom_model.py : contains several models. The Net16 model retrieves the pre-entrained weights of the first part of the layers of the crack_segmentation model. 
Link to the repo: https://github.com/khanhha/crack_segmentation. The model is located in unet/unet_transfer.py. The weights are downloadable in the inference part of the README associated to this REPO : unet_vgg16.  
One way is to use the masks produced by this repo as data instead of the frames extracted from the pipe videos.
