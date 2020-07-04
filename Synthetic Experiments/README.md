# Discriminative Feature Alignment for Digit Classification
##### A prior-guided latent alignment approach for Unsupervised Domain Adaptation

This is the code implementation of the synthetic experiments to validate the distribution alignment mechanism of our proposed ```distribution alignment loss``` in Pytorch. The code is implemented by Jing Wang.

The experimental results are shown below

![align](align.png)

# Getting Started

#### Installation

* Install PyTorch ```pip install torch```
* Install Numpy ``` pip install numpy ```
* Install matplotlib ``` pip install matplotlib ```
* Install sklearn ``` pip install scikit-learn ```

# Dataset

If you want to test the distribution alignment mechanism on 2D Gaussian Blobs dataset, please revise ```line 6``` in ```moon.py```.

# Train

* Here is an example for running the experiment on moon dataset and 2D Gaussian dataset:

``` python moon.py``` 
``` python 2d_G.py``` 
