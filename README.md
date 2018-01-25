# Wide Residual Networks with Residual Adapter Modules in Keras

Implementation of wide residual network with Residual Adapter Module from the paper <a href="https://arxiv.org/abs/1705.08045">Learning multiple visual domains with residual adapters</a> in Keras.

This is ongoing work on the implementation of the paper 'Learning multiple visual domains with residual adapters'. 

## Objective: 
make a single neural network that is good at classifying images from multiple visual domains - specifically, 10 visual domains, called the decathlon challenge.

## Approach:
This leads to a model that is 2X the size of the individual domain network, with 90% of the parameters shared between domains and 10% of the parameters are domain specific.
Evaluation:

Baseline error is given as twice the error for each domain for a model with 10x parameters - 10 neural networks, fully finetuned on each domain. This has an s-score of 2500.
The RAM model gives a score of 2118 when decay coefficients are not optimized for each domain, but gives a superior score of 2643 points when higher decay coefficient (0.002 or 0.005 rather than the default 0.0005) is used for the domains with less data.

## Claims:
The key claim is that the performance of the 2X size model with residual adapters proposed here has similar or better effectiveness on the visual decathlon task than the 10x sized fully finetuned model. The proposed model is smaller and takes less time to converge.

The paper claims to have 0.1x domain specific parameters for each network. In the present implementation there are a total of 6,558,770 parameters of which 704,194 are domain specific. This is ~0.1x of the total parameters, in line with the paper.

Domain specific trainable parameters include:

1. all batch norms
2. all convolutions within the residual adapters.

The frozen parameters are all the convolutions not part of the residual adapters.


## Models
The below model is the WRN-28-4-RAM model implemented here. However, this is not an exact match with the paper and does not train well on imagenet in its current form. 
<img src="https://github.com/astha-chem/Decathlon-Residual-Adapters-Keras/blob/master/plots/WRN-28-4-RAM.png" height=100% width=100%>
