# Neural Network

Neural network implementation written in plain Scala. No third-party ML libraries are used.

It is supposed to be used only for educational purposes so no performance optimization is done.  
The network and neurons are implemented in pretty straightforward (and non-optimal) way.

## Scope

### Done

- The [Feed-forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) algorithm to predict labels
- Simple [perception](https://en.wikipedia.org/wiki/Perceptron) with step activation function
- A sample network to solve [XOR](https://www.google.nl/?ion=1&espv=2#q=xor%20neural%20network) based on perceptions
- Additional activation functions, like [Identity](https://en.wikipedia.org/wiki/Identity_function) and [Sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)
- The [Back-propagation](https://en.wikipedia.org/wiki/Backpropagation) algorithm to train network
- [Stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent) algorithm
- Multiple loss (cost) functions, like [Quadratic](https://en.wikipedia.org/wiki/Loss_function#Quadratic_loss_function) and [Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy)
- Simple [Linear Regression](https://en.wikipedia.org/wiki/Linear_regression) examples
- Add [regularization](https://www.quora.com/What-is-regularization-in-machine-learning)
- Apply [dropouts](https://www.quora.com/How-does-the-dropout-method-work-in-deep-learning)
- Build and train a network on [MNIST](http://yann.lecun.com/exdb/mnist/) data
