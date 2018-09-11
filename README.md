# Identify plant species from pictures

This repo is a demo of a cnn in pure tensorflow ( almost no keras except for image preprocessing).
I did this as an exercise to use a low level machine learning framework and have a good understanding of how the underlaying layer i usualy use with keras works.

I trained several simple Cnn from scratch on a kaggle dataset.

I implemented 3 differents models for now.

the first is a simple CNN.
The second is a CNN with batch normalization
On the third one i added another CNN layer and modified the loss in order to take into account class unballence
I obtain 93% of accuracy which is decent for a simple model without transfer learning.

## Run

download the dataset from kaggle with kaggle-api, unzip train.zip and test.zip

```sh
git clone
cd plant-seedlings-classification
python train.py -p <path of data> -m <version of the model to use from 1 to 3> -s <image resized to s * s>
```
