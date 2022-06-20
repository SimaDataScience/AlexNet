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
### Optimization
AlexNet utilizes stochastic gradient descent with momentum and constant weight decay, with values of 0.9 and 0.0005 respectivley. The authors note that weight decay was not simply a means of regularization, but was a requirement for their model to learn. Additionally, an adaptive learning rate is used. The learning rate is initialized to 0.01 and is decreased by a factor of 10 when validation error stops decreasing. The exact criteria for this shrinking are not specified by the authors, so this implementation uses a fairly standard heuristic.

### Overlapping Maxpooling Layers
AlexNet contains three maxpooling layers following the first, second, and fifth convolutional layers. A kernel size of 3x3 and a stride of (2, 2) is used to intentionally overlap these maxpooling layers, which further increases the model's performance.

### Local Response Normalization
While batch normalization is now the most prevelant layer normalization method for avoiding vanishing and exploding gradients when training very deep networks, AlexNet instead uses local response normalization. This method has largely been replaced by batch normalization, but our model still utilizes local response normalization to stay true to the original paper.

### Data Augmentations
AlexNet employs three main types of data augmentation. The first form of data augmentation is to both train and predict on 224x224 slices of our images, rather than the original 256x256 image, which allows us to significantly artificially increase the size of our dataset. The second method of data augmentation is to simply reflect the image horizontally. Lastly, PCA analysis is performed over the entirety of our RGB values and the PCA terms are used to transform our original RGB values.

While the set of data augmentations is identical during both training and prediction, the method that they are applied is slightly different during these phases. Firstly, our model calculates the PCA terms needed for augmentation upon model initialization. Next,during training, each input image is randomly flipped horizontally with probability 0.5, and then a random 224x224 slice of the image is then chosen. The RGB values are then transformed, and image is sent into our network. Finally, upon prediction, each 256x256 input image is used to create 10 new images on which to predict: the corners and center of our original image create 5 of our training images, and their reflections create the remaining 5. The PCA values of these 10 images are then transformed, and then predictions are made across these 10 new images. Our final prediction is then the average across these 10 predictions.

### Additional Implementation Details
The paper includes many finer details that are less novel. Relu activation is used for non-output layers, weight and bias initializations are specified, etc. Each of these details are accounted for, but are likely of little interest.

## Deviations From the Paper
The creators of the AlexNet network trained their model across two GPUs by splitting the networks layers between the GPUs, with layers from separate GPUs being combine only sporadically. This implementation takes a different approach to multi-GPU utilization by instead giving the options to split the entire training task across multiple GPUs. Rather than each GPU being responsible for only a subset of the entire model, this implementation utilizes Tensorflow's MirroredStrategy to fully distribute model training across all available GPUs on the current machine. This can be easily extended to train on multiple workers using Tensorflow's MultiWorkerMirroredStrategy if desired.

## Example Implementation


## Source
Thanks for checking out my repo. For more information please see the original paper here: https://papers.nips.cc/paper/2012/hash/c399862d3b9d6b76c8436e924a68c45b-Abstract.html
