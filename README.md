# AlexNet
Implementation of AlexNet Network

## Basic Architecture
The AlexNet architecture consists of five convolutional layers, and three fully connected layers.
Three convolutional layers (layers 1, 3, and 5) are followed by overlapping maxpooling layers,
and the first and second maxpooling layers are followed by local response normalization layers.
While ImageNet contains 256x256 color images, the model is trained on 224x224 slices of these images,
in order to construct additional samples.
Furthermore, predictions are made by creating 10 such 224x224 slices (method described below) from our original 256x256 image,
and averaging predictions across these slices.

## Model Construction Details

## Deviations From the Paper

## Example Implementations

## Source
