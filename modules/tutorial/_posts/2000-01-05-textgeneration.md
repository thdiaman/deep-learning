---
title: Text Generation
---

# Text Generation

## Problem Description and Data

The Text Generation problem uses the bAbi dataset, of which an excerpt can be downloaded
<a target="_blank" href="{{site.dataurl}}/TextGeneration/qa1_single-supporting-fact_train.txt">here (training set)</a> and
<a target="_blank" href="{{site.dataurl}}/TextGeneration/qa1_single-supporting-fact_test.txt">here (test set)</a>. Use
<a target="_blank" href="{{site.dataurl}}/TextGeneration/stories.py">this script</a>
for loading and manipulating the dataset.
Save the dataset in a folder and let's start coding.

Check also <a target="_blank" href="{{site.baseurl}}/presentations/TextGeneration.pdf">here</a>
for a presentation on this problem.

## Problem Solution

### Solution using Recurrent Neural Network (LSTM)

- <a target="_blank" href="{{site.dataurl}}/TextGeneration/training_rnn.py">Training a Recurrent Neural Network</a>
- <a target="_blank" href="{{site.dataurl}}/TextGeneration/rnn_model.h5">Trained Model</a>
- <a target="_blank" href="{{site.dataurl}}/TextGeneration/dictionary.json">Vocabulary</a> and
<a target="_blank" href="{{site.dataurl}}/TextGeneration/lengths.json">lengths</a>
- <a target="_blank" href="{{site.dataurl}}/TextGeneration/testing_rnn.py">Testing the Model</a>

