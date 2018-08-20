# Info
This repository contains the code for the paper "Accurate and Diverse Sampling of Sequences based on a “Best of Many” Sample Objective" (https://arxiv.org/abs/1806.07772). 
Currently only the code corresponding only to the experiments with the MNIST Sequence Dataset is available. Will be expanded in the near future.

# Requirements
The code was tested on,
* Tensorflow - 1.3.0
* Keras - 2.1.5

Aditional requirements,
* Matplotlib for plotting.
* Tqdm for progress bars.

# Usage
The MNIST Sequence dataset can be found at: (https://edwin-de-jong.github.io/blog/mnist-sequence-data/). Please download and extract the sequences and place them under `./data/MNIST/sequences/`.

To train a CGM model using the "Best of Many" objective use,

`python mnist.py`

The (negative) CLL and the KL divergence between the true and learned latent distributions is printed.

To train without plotting use,
`python mnist.py --plot False`

# Citation
@inproceedings{apratim18cvpr2,  
title = {Accurate and Diverse Sampling of Sequences based on a {\textquotedblleft}Best of Many{\textquotedblright} Sample Objective},  
author = {Bhattacharyya, Apratim and Fritz, Mario and Schiele, Bernt},  
year = {2018},  
booktitle = {31st IEEE Conference on Computer Vision and Pattern Recognition (CVPR 2018)},  
}
