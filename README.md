# Deep Multi-Object Tracking

Multiple Object Tracking Using Deep Learning and Kalman Filter. 

The tracker implemented combine the following two papers:
1. https://arxiv.org/pdf/1701.01909.pdf
2. https://arxiv.org/abs/1602.00763

The first paper present a state of art deep learning model to do multiple object tracking.
The second paper present a simple online Kalman Filter tracker.

The model implemented in this repo novelly combine both tracker, which can do online multiple object tracking using a state of art deep learning model to identify the apperance features and a simple online Kalman filter tracker to identify motion features.

## Overview of Files

### Deep Apperance Model
*   `data_input.py`: Script to preprocess the MOT15 data.
*   `train.py`: Contains training process of the apperance network.
*   `apperance_network.py`: Contains the deep apperance model architecture.


### Kalman Filter Tracking
*   `kalman_filter_tracker.py`: Taken from the implementation of the original author.


### Deep Tracking
*   `deep_tracker.py`: A new class which novelly combine both trackers to consider both apperance and motion featuers.
