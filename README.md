# Generate synthetic historical maps from OSM map tiles

We use cycle-GAN to convert open street map (OSM) images into historical map style. The historical map dataset used for training is downloaded from National Library of Scotland ([NLS](https://maps.nls.uk/)) website. 

- For training, run cycleGAN-keras.py
- For generating synthetic historical maps with trained model, run sytle_convert.py

## Live Demo
A live demo is available at https://zekun-li.github.io/side-by-side/

This demo provides two radio buttons to toggle between two views. The first view compares the synthesized historical map with the input OSM map, and the second view compares the synthesize map with authentic historical maps.

![screencast example](demo.gif)


## Pretrained Weights
Weight files can be downloaded [here](https://github.com/tjwei/GANotebooks/blob/master/CycleGAN-keras.ipynb)

## Generated Synthetic Historical Map Tiles
Synthetic historical map tiles can be downloaded from Google Drive at [here](https://github.com/tjwei/GANotebooks/blob/master/CycleGAN-keras.ipynb)

Ackowlegment:
The code is adapted from [this](https://github.com/tjwei/GANotebooks/blob/master/CycleGAN-keras.ipynb) github repo.
