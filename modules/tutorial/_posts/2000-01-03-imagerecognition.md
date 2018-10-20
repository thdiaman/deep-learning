---
title: Image Recognition
---

# Image Recognition

## Problem Description and Data

The Image Recognition problem uses the Cats vs Dogs dataset, which can be downloaded
<a target="_blank" href="{{site.dataurl}}/ImageRecognition/data.zip">here</a>.
Save the dataset in a folder and let's start coding.

Check also <a target="_blank" href="{{site.baseurl}}/presentations/ImageRecognition.pdf">here</a>
for a presentation on this problem.

## Problem Solution

### Solution using Convolutional Neural Network

- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/training.py">Training a Convolutional Neural Network</a>
- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/model.h5">Trained Model</a>
- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/testing.py">Testing the Model</a>

### Solution using Bottleneck Features on VGG16

- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/vgg16_pretrained_imagenet.h5">Pretrained VGG16</a>
- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/training_vgg16.py">Training the Network</a>
- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/bottleneck_features.zip">Extracted Bottleneck Features</a>
- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/bottleneck_fc_model.h5">Trained Model</a>
- <a target="_blank" href="{{site.dataurl}}/ImageRecognition/testing_vgg16.py">Testing the Model</a>

## Exercise
Can we perform the same analysis for multi-class Image Recognition?

Use the excerpt from the CIFAR10 dataset that is available
<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/data.zip">here</a>.

Try this before you check the solution
(<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/training.py">training</a>,
<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/model.h5">trained model</a>, and
<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/testing.py">testing</a>)
and the VGG16 solution
(<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/training_vgg16.py">training</a>,
<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/bottleneck_features.zip">bottleneck features</a>,
<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/bottleneck_fc_model.h5">trained model</a>, and
<a target="_blank" href="{{site.dataurl}}/ImageRecognitionMultiClass/testing_vgg16.py">testing</a>).
