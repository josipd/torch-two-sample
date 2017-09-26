# `torch-two-sample`
[![Documentation Status](https://readthedocs.org/projects/torch-two-sample/badge/?version=latest)](https://torch-two-sample.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.org/josipd/torch-two-sample.svg?branch=master)](https://travis-ci.org/josipd/torch-two-sample)

A PyTorch library for differentiable two-sample tests

### Description

This package implements a total of six two sample tests:

  * The classical Friedman-Rafsky test *[FR79]*.
  * The classical k-nearest neighbours (kNN) test *[FR83]*.
  * The differentiable Friedman-Rafsky test *[DK17]*.
  * The differentiable k-nearest neighbours (kNN) test *[DK17]*.
  * The maximum mean discrepancy (MMD) test *[GBR+12]*.
  * The energy test *[SzekelyR13]*.

Please refer to the [documentation](https://torch_two_sample.readthedocs.io)
for more information about the project.
You can also have a look at the following [notebook](notebooks/mnist.ipynb)
that showcases how to use the code to train a generative model on MNIST.

### Installation

After installing PyTorch, you can install the package with:

```
python setup.py install
```

### Testing

To run the tests you simply have to run:

```
python setup.py test
```

Note that you will need to have [Shogun](http://www.shogun-toolbox.org)
installed for one of the test cases.


### Bibliography

  * *[DK17]* J. Djolonga and A. Krause. Learning Implicit Generative Models Using Differentiable Graph Tests. ArXiv e-prints, September 2017. arXiv:1709.01006.
  * *[FR79]* Jerome H Friedman and Lawrence C Rafsky. Multivariate generalizations of the wald-wolfowitz and smirnov two-sample tests. Annals of Statistics, pages 697–717, 1979.
  * *[FR83]* Jerome H Friedman and Lawrence C Rafsky. Graph-theoretic measures of multivariate association and prediction. Annals of Statistics, pages 377–391, 1983.
  * *[GBR+12]* Arthur Gretton, Karsten M Borgwardt, Malte J Rasch, Bernhard Schölkopf, and Alexander Smola. A kernel two-sample test. Journal of Machine Learning Research, 13(Mar):723–773, 2012.
  * *[SST+12]* Kevin Swersky, Ilya Sutskever, Daniel Tarlow, Richard S Zemel, Ruslan R Salakhutdinov, and Ryan P Adams. Cardinality restricted boltzmann machines. In Advances in Neural Information Processing Systems (NIPS), 3293–3301. 2012.
  * *[SzekelyR13]* Gábor J Székely and Maria L Rizzo. Energy statistics: a class of statistics based on distances. Journal of Statistical Planning and Inference, 143(8):1249–1272, 2013.
  * *[TSZ+12]* Daniel Tarlow, Kevin Swersky, Richard S Zemel, Ryan Prescott Adams, and Brendan J Frey. Fast exact inference for recursive cardinality models. Uncertainty in Artificial Intelligence (UAI), 2012.
