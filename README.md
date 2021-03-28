# ANN
Implementation from scratch of an artificial neural network. As a proof of concept, a model was further created for the "10 Monkey Species" classification problem, along with an analysis of results. 

## Problem Anlysis
The goal of the current project is to implement from scratch an Artificial Neural Network that
can be used for multi-class image classification. The chosen type of ANNs is Feed Forward
Network. The data that will be classified is the “10 Monkey Species" dataset, consisting of
images of 10 different monkey species.

### Data
The dataset images are part of 10 different classes, with a total of 1370 images. Below, we can
observe an image from each class. The displayed images allow distinctive features of each
species to be easily retrieved, but the dataset contains some challenging images that make the
task difficult.

![alt text](dataset_sample.png)

The table below illustrates the number of images per class. We notice that the dataset is balanced.
| Class 0 | Class 1| Class 2 | C;ass 3 | Class 4 | Class 5 | Class 6 | Class 7 | Class 8 | Class 9|
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 131 | 139 | 137 | 152 | 131 | 141 | 132 | 142 | 133 | 132 |
