# Letter_Recognition----TensorFlow
Using TensorFlow for letter classification by accessing properties of black-and-white, rectangular pixels.
I used 2 hidden layers, applying a Sigmoid and Softmax activation fuction in both layers, respectively.
For backpropogation, I used the standard Cross-Entropy loss fuction, which provides a probality for each classified letter.
The Backpropogation method I used was Gradient Decent.
I tested numerous values for the hyperparmeters, with the values I finally used being a learning-rate = 0.01 and Epochs = 100.
