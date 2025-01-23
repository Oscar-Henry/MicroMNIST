This project implement the Mini Grad repository developped by Andrej Karpathy with the MNIST data set.

The MNIST data set is constitued of 70 000 black and white images of number in [0, 9].
The training set will be constituted of 60 000 images.
The test set will be constituted of 10 000 images.

The images are in dimension (1, 28, 28).

We generate a MLP model with 784 input neurons, 2 hidden layers of 16 neurons and a final layer of 10 neurons to compute the probability of each label.