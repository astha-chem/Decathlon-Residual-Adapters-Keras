# Wide Residual Networks with Residual Adapter Modules in Keras

Implementation of wide residual network with Residual Adapter Module from the paper <a href="https://arxiv.org/abs/1705.08045">Learning multiple visual domains with residual adapters</a> in Keras.

Here's a summary of the paper: 

## Objective: 
make a single neural network that is good at classifying images from multiple visual domains - specifically, 10 visual domains, called the decathlon challenge.

## Approach:
This leads to a model that is 2X the size of the individual domain network, with 90% of the parameters shared between domains and 10% of the parameters are domain specific.

## Evaluation:
Baseline error is given as twice the error for each domain for a model with 10x parameters - 10 neural networks, fully finetuned on each domain. This has an s-score of 2500.
The RAM model gives a score of 2118 when decay coefficients are not optimized for each domain, but gives a superior score of 2643 points when higher decay coefficient (0.002 or 0.005 rather than the default 0.0005) is used for the domains with less data.

## Claims:
The key claim is that the performance of the 2X size model with residual adapters proposed here has similar or better effectiveness on the visual decathlon task than the 10x sized fully finetuned model. The proposed model is smaller and takes less time to converge.

The paper claims to have 0.1x domain specific parameters for each network. In the present implementation there are a total of 6,558,770 parameters of which 704,194 are domain specific. This is ~0.1x of the total parameters, in line with the paper.

Domain specific trainable parameters include:

1. all batch norms
2. all convolutions within the residual adapters.

The frozen parameters are all the convolutions not part of the residual adapters.

## Dataset:
The dataset for 9 of the 10 domains (all except imagenet), resized to 72 px on the shorter dimension can be downloaded here (406 MB): http://www.robots.ox.ac.uk/~vgg/decathlon/

The imagenet dataset can be downloaded through the image-net.org website (6.1 GB)

## Model
I have implementated the ResNet28_RAM model in Keras, tested with TensorFlow backend and Python 3.6. I would like to thank the author Sylvestre-Alvise Rebuffi for kindly sharing snippets of his code in PyTorch before it was publically available, that aided in implementing this model. The architecture is a mix of the classical ResNet and the Wide Residual Networks. 

Possible differences from the paper: 
1. L2 Regularization: The paper does not explicitly mention regularization, but in this implementation L2 regularization was used with a value 0.0001. Without regularization, training accuracy could go up to 100 % for SVHN, but validation accuracy stalled at <40%. 
2. Scaling down dimensions at each block: The authors do not discuss this in the paper explicitly. The architecture consists of 3 blocks, each of which contains 4 residual units. The dimensions for these blocks are 32x32x64, 16x16x128 and 8x8x256 (starting with 64x64x3 images). The dimensions scale down at each block. I have used a 1x1 convolutional shortcut for this, as in the Wide Residual Networks implementation, while the authors did this in 2 steps : 1. (2,2) average pooling, and concatenate the tensor with a '0' tensor of the same dimensions, along the channel axis. 

This is a schematic of the model: 
<img src="https://github.com/astha-chem/Decathlon-Residual-Adapters-Keras/blob/master/plots/ResNet28_RAM.png" height=100% width=100%>

## Training
The model with Residual Adapter Modules was trained from scratch on the SVHN dataset. The dataset has 70k images and 10 classes. 
The paper reports accuracy of 96.63% for this dataset. The best accuracy obtained in my tests was 93.6% (validation accuracy). 

Training was done using plain SGD without momentum, the learning rate was gradually reduced from 0.1 to 0.01 and 0.001, with each being for 5-6 epochs. Batch size was 64, and 64x64 images were used. 

The training time was ~380s per epoch on a Maxwell architecture GPU with 1664 CUDA cores and 8GB memory (Paperspace GPU+).





