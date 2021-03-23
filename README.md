# Discriminative Feature Alignment
##### A prior-guided latent alignment approach for Unsupervised Domain Adaptation

![Idea](overallidea.png)

This is the code implementation of Discriminative Feature Alignment for digit and object classification in Pytorch. The code is implemented by Jing Wang. If you have any questions or find any mistakes in the code implementation, please do not hesitate to email me at jing@ece.ubc.ca

Paper Name: Discriminative Feature Alginment: Improving Transferability of Unsupervised Domain Adaptation by Gaussian-guided Latent Alignment [[link to Paper (ArXiv)]](https://arxiv.org/abs/2006.12770) [[Link to Paper (Pattern Recognition)]](https://www.sciencedirect.com/science/article/pii/S0031320321001308?casa_token=Yg5kXSnM-ycAAAAA:U5cNbRVY25-imG02vTBVwE7MYvRHU08IbmFsYaWHZlyCVzqSgyFqd3k8a-sBAuQitgkjtKmRJQ)

One of the key contribution of this paper is introducing a new method to align any two distributions, which is GAN explored. Instead of optimizing the discriminator error, it minimizes the direct L1-distance between the decoded samples in the feature space.

![alignment](alignment.png)

Below are the results that can validate the distribution alignment mechanism of our proposed regularization:

![experiment](experimentForAlignment.png)

# Instructions

#### The instructions for the experiments are inside the directories ***Digit_Classification*** and ***Object_Classification***. 


# Citation

Please cite our paper if you use our code for your work.
```
@article{wang2021discriminative,
  title={Discriminative Feature Alignment: Improving Transferability of Unsupervised Domain Adaptation by Gaussian-guided Latent Alignment.},
  author={Wang, Jing and Chen, Jiahong and Lin, Jianzhe and Sigal, Leonid and de Silva, Clarence W},
  journal={Pattern Recognition},
  pages={107943},
  year={2021},
  publisher={Elsevier}
}
```

# References

[MCD_DA](https://github.com/mil-tokyo/MCD_DA)

[AFN](https://github.com/jihanyang/AFN)
