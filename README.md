# Micro Grad for categorical classification

This project implement the Micro Grad pytorch API-like repository developped by Andrej Karpathy completed with the softmax and Cross-Entropy loss function and its backpropagation allowing to work with categorical classification problems.
The MLP class allow the construction of NN using DAG-like structures.

![DAG](images/DAG.png)

### Example usage

The two notebook:
`demo_IRIS.ipynb`
`demo_MNIST.ipynb`

provides a fulld demo of training a 2-layer NN (MLP) categorical classifier. Including the loading of the datasets, the initialization of a NN module for `micrograd.nn`, implementing a "Cross-Entropy" categorical classification loss function and using SGD for optimization.
As shown in the notebook, with the IRIS dataset, using aa 2-layer neural net with one 5-nodes hidden layer and a 3-node output layer we achieve nearly perfect prediction on the test set:
`Loss = 0.111, accuracy = 96.667%`

### License

MIT