# WGAN-GP on the Billion Words Dataset

I could not find a Pytorch implementation for the character level language modeling task presented in [1], so I ported it into PyTorch. 

## Usage

* Download the 1 Billion Words dataset from [here](http://www.statmt.org/lm-benchmark/) and extract in `./data/`.
* Execute `wgan_text.py`.
* The results and models will be saved in `./results/`.


### Acknowledgments

This code has been ported from the original code [igul222/improved_wgan_training](https://github.com/igul222/improved_wgan_training) which was released under MIT License.

[1] Gulrajani, Ishaan, et al. "Improved training of Wasserstein GANs." Advances in neural information processing systems. 2017.
