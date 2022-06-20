# AlexNet
## Implementation of AlexNet Network
The following is an implemenation of AlexNet for image classification. It allows you to train a model with identical specifications to AlexNet on training and validation images saved in a specified directory (local or not).

## Basic Architecture
The AlexNet architecture consists of five convolutional layers, and three fully connected layers.
Three convolutional layers (layers 1, 3, and 5) are followed by overlapping maxpooling layers,
and the first and second maxpooling layers are followed by local response normalization layers.
While ImageNet contains 256x256 color images, the model is trained on 224x224 slices of these images,
in order to construct additional samples.
Furthermore, predictions are made by creating 10 such 224x224 slices (method described below) from our original 256x256 image,
and averaging predictions across these slices.

## Model Construction Details
Weight/bias initializations, overlapping pooling, activations, local response over batch normalization, optimization algorithm,
image augmentations (slicing, flipping, PCA), prediction augmentations.

## Deviations From the Paper
Entire network split between GPUs instead of specific layers.

## Example Implementations

## Source
https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
