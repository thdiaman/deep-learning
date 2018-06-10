---
title: Text Classification
---

# Text Classification

## Problem Description and Data

The Text Classification problem uses the Twitter Sentiment dataset, which can be downloaded
<a target="_blank" href="{{site.dataurl}}/TextClassification/15000tweets.csv">here</a>.
Save the dataset in a folder and let's start coding.

Check also <a target="_blank" href="{{site.baseurl}}/presentations/TextClassification.pdf">here</a>
for a presentation on this problem.

## Problem Solution

### Solution using Fully Connected Neural Network

- <a target="_blank" href="{{site.dataurl}}/TextClassification/training.py">Training a Fully Connected Neural Network</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/model.h5">Trained Model</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/dictionary.json">Vocabulary</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/testing.py">Testing the Model</a>

### Solution using Convolutional Neural Network

- <a target="_blank" href="{{site.dataurl}}/TextClassification/training_cnn.py">Training a Convolutional Neural Network</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/cnn_model.h5">Trained Model</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/dictionary.json">Vocabulary</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/testing_cnn_lstm.py">Testing the Model</a>

### Solution using Recurrent Neural Network (LSTM)

- <a target="_blank" href="{{site.dataurl}}/TextClassification/training.py">Training a Recurrent Neural Network</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/lstm_model.h5">Trained Model</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/dictionary.json">Vocabulary</a>
- <a target="_blank" href="{{site.dataurl}}/TextClassification/testing_cnn_lstm.py">Testing the Model</a>

## Exercise
Can we perform the same analysis for multi-class Text Classification?

Use the excerpt from the BBC news dataset that is available
<a target="_blank" href="{{site.dataurl}}/TextClassificationMultiClass/bbc.zip">here</a>
and can be loaded using <a target="_blank" href="{{site.dataurl}}/TextClassificationMultiClass/dataload.py">this script</a>.

Try this before you check the solution
(<a target="_blank" href="{{site.dataurl}}/TextClassificationMultiClass/training.py">training</a>,
<a target="_blank" href="{{site.dataurl}}/TextClassificationMultiClass/model.h5">trained model</a> and
<a target="_blank" href="{{site.dataurl}}/TextClassificationMultiClass/dictionary.json">vocabulary</a>, and
<a target="_blank" href="{{site.dataurl}}/TextClassificationMultiClass/testing.py">testing</a>).

