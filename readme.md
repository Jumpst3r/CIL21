# Baseline I
## This branch implements a simple first baseline:

### How
We generate a training set by cutting out 20x20 patches from the provided images and labelling them wither as roads (1) or non-roads (0).

This behavior is implemented in the file [`feature_extractor.py`](feature_extractor.py)

We then load this data and train a very simple CNN (4 convolutional layers, 2 fully connected layers) on the generated patches.
This behavior is implemented in the file [`baseline_training.py`](baseline_training.py), which also saves the trained model weights as a `.ckpt` file

We use the F1 metric to asses the quality of our model during training on the validation set, implemented in [`f1_score.py`](f1_score.py).

Finally the file [`predict.py`](predict.py) can be used to vizualize the predictions on the test data.

### Downsides

As this is supposed to be a simple baseline, here are a few obvious downsides:

- Classification is on a 20x20 patch level, very coarse
- No context, neighbouring patches do not change current patch predictions

### How to play with this baseline:

Clone the branch: `git clone --branch baseline-I git@github.com:Jumpst3r/CIL21.git`

Create & activate a new python environment: `virtualenv env && source ./env/bin/activate`

Install required packages : `pip install -r requirements.txt`

Create a `patches` directory: `mkdir patches`

Run the patch generation script: `python3 feature_extractor.py`

Run the training script: `python3 baseline_training.py`

Visualize predictions: `python3 predict.py` (make sure to use the latest weights file)