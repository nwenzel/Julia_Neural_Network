# Code for Basic Julia Neural Network #

This repository contains all the Julia code for a basic Neural Network that uses full-batch back propagation to learn feature weights for each layer in the network architecture. The code will allow you to create networks with any number of hidden layers each with any number of units. Input layer and Output layer units are determined by training data.

The easiest way to run it is to call XOR_Test([2], 2500) which creates a 3 layer net with a 2 unit hidden layer that will run 100 iterations over the full set of training data. That test takes about 1 second to run on my machine (Mac Book Air 1.7GHz i5 processor 4GB RAM).

The code works with any number of hidden layers and any number of units in each layer, so go crazy trying out different combinations. BUT... it has not been tested on large data sets, large numbers of hidden layers, or large numbers of units in the hidden layer. So... go crazy at your own risk.

## Important Note ##
Neural networks that use back propagation will sometimes fail due to the back prop algorithm landing in a local minimum. From time to time, this algorithm will fail. So now you know.

## Give it a try ##
It's pretty easy to get in and mess with the code. It's all contained in a single file to make it easy to try out.

initialize_theta(2, 1, [2]) would create a list of Theta terms with dimensions 2X3 and 1X3.
initialize_theta(6, 3, [4, 4]) would create a list of Theta terms with dimensions 4X7, 4X5, and 3X5


####At the Julia prompt:####

load("julia_nn_function.jl")

XOR_Test([2], 2500)
XOR_Test([2], 5000)
XOR_Test([2], 10000)
XOR_Test([2,2], 2500)
XOR_Test([2,2,2], 2500)  <-- This will run and the math will work, but it won't be correct
XOR_Test([10], 2500)  <-- This will work but is ridiculous


####At the Terminal:####
Make changes to the last line of the file to adjust the architecture. Then...

julia julia_nn_function.jl


The output should show the training data (all 4 records), the classes of the training data, an explanation of the architecture, Theta shapes and values, the cost of the last iteration of the back prop function, a new set of input data to test, and predicted outputs.


From there, go in and mess with the code.

### Requirements ###
Julia
Version 0.0.0+99580215.r6242

## Author ##

 - Nathan Wenzel, Edge Solutions, Inc. [http://www.edgesolutions.com/](http://www.edgesolutions.com/)

[@nwenzel](http://twitter.com/nwenzel)

## License ##

All source code is copyright (c) 2012, under the Simplified BSD License.  
For more information on FreeBSD see: [http://www.opensource.org/licenses/bsd-license.php](http://www.opensource.org/licenses/bsd-license.php)

All images and materials produced by this code are licensed under the Creative Commons 
Attribution-Share Alike 3.0 United States License: [http://creativecommons.org/licenses/by-sa/3.0/us/](http://creativecommons.org/licenses/by-sa/3.0/us/)
