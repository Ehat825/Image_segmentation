# Image_segmentation Project
The goal of this project was to create a decoder layer for an existing object recognition neural network.

The data used was was pascal VOC, which includes images and segmentations masks for 21 different categories, since these images are of varying size, the main data preprocessing was forcing them all to be 256 by 256.

The libraries used were pytorch, matplotlib, numpy and torchvision.
I used Claude Sonnet to generate a lot of the actual code.
I ended up testing two different decoder layers, one using bilinear upsampling with convolutions, and another that uses purely transposed convolutions.
Overall, their performance seems pretty similar, with bilinear doing a bit better. The quality of the output masks is pretty similar, with the blinear model outputting more "solid" looking boxes 
while the transposed deconvolutional model tends to have "dustier" (for lack of a better term) object boundaries.

The comparative analysis of the two selects the category which they did best in relative to eachother, worst overall, and the class with the most average performance. The metric used for this was intersection over union.
Across repeated runs which classes are the best/average/worst seems to change,on the saved version on this github for instance, one of the models did best on the "background" class

In order to repeat my results you need to upload the notebook to a colab environment with a GPU and run it, be aware training usually takes about 30 minutes.

Ultimately time and compute for running these models was a big roadblock, I often ran out of my allowance on colab, which hindered progress and made me not want to mess with it too much.

If I kept going with this project I'd like to try data augmentation, more types of decoder layers, adding toggleble skip connections as well as adding more hyperparameters. I never managed to get it to play nice with weights and biases.
