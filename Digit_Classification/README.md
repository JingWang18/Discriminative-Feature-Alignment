# Discriminative Feature Alignment for Digit Classification
##### A prior-guided latent alignment approach for Unsupervised Domain Adaptation

This is the code implementation of Discriminative Feature Alignment for digit and object classification in Pytorch. The code is implemented by Jing Wang.

The t-SNE visualization of our method is shown below

![tsne](tsne.png)

# Results

| Method  | SVHN-MNIST | SYNSIGNS-GTSRB | MNIST-USPS | USPS-MNIST |
| ------------- | ------------- | ------------- | ------------- |------------- |
| MCD  | 96.2  | 94.4 | 96.5 | 94.1 |
| DFA-ENT (***Ours***)  | 98.2 | 96.8 | 97.9 | 96.2 |
| DFA-MCD (***Ours***)  | 98.9 | 97.5 | 98.6 | 96.6 |

# Getting Started

#### Installation

* Install PyTorch and its dependencies ```pip install torch torchvision```
* Install torchnet ```pip install git+https://github.com/pytorch/tnt.git@master```

# Dataset

Download MNIST data from [here](https://drive.google.com/file/d/1cZ4vSIS-IKoyKWPfcgxFMugw0LtMiqPf/view). You download other datasets from their official websites. Please create a folder named "data" and put the dataset in the directory ./data.

# Train

* Here is an example for running experiment on the adaptation scenario from SVHN to MNIST:

``` python main.py --source svhn --target mnist```
