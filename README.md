# Classifying Images with Neural Networks
Computer vision experimentation using ANNs and CNNs to classify different types of images.


### [1] Description
This repository consists of two computer vision projects. The first project uses artificial neural networks to classify grayscale Japanese handwritten Kanji characters. The second project uses advanced convolutional neural networks to classify coloured images of objects and animals.


### [2] Use Case
These projects enable a computer to easily classify images and match them to their true contents with a relatively high accuracy.


### [3] Repository Contents
1) ANN Image Classifier.ipynb<br>
This classifier is built based on the k_mnist dataset, which is available at  https://www.tensorflow.org/datasets/catalog/kmnist. More details about this dataset are here: https://github.com/rois-codh/kmnist. For this classifier, I am limiting myself to only using dense ANNs rather than advanced CNNs. The Kuzushiji-MNIST dataset contains 70,000 28x28 grayscale images spanning 10 classes of Japanese characters.

2) CNN Image Classifier (Transfer Learning).ipynb<br>
This classifier was built based on a pre-trained network (MobileNetV2) to do image classification on the CIFAR-10 dataset. More information on the dataset is available here:  https://www.cs.toronto.edu/~kriz/cifar.html. For this classifier, I explore the use of transfer learning in the context of CNNs. The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.


### [4] Author
Martin Cheng


### [5] References
1) **Deep Learning for Classical Japanese Literature. Tarin Clanuwat et al. [arXiv:1812.01718](https://arxiv.org/abs/1812.01718)**
2) **[Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009.**
