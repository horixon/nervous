Nervous
=========

Cross platform Neuralnet

##Tools
  - Neuralnet - forward and backward propagation
  - Logistic - sigmoid

##Platforms
  - Core neuralnet in c/c++
  - iOS

###Neural Net
Neuralnet with forward and backward propagation.  Implemented iOS using [Accelerate Framework](https://developer.apple.com/library/mac/documentation/Accelerate/Reference/AccelerateFWRef/_index.html).  Net training with [steepest decent](http://en.wikipedia.org/wiki/Gradient_descent).  Theta is the bias and weights.  Matrices are expected in [column major](http://en.wikipedia.org/wiki/Row-major_order#Column-major_order) order, columns are continuous in memory.

Example of usage [here](https://github.com/horixon/data-tools/blob/master/Neuralnet/NeuralnetTests/NeuralnetTests.m).  The consumer of Neuralnet.m is responsible for handling the memory allocation.

#####References
[Pattern Classification](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0471056693.html)  
[Neural Networks:Tricks of the Trade](http://rd.springer.com/book/10.1007/978-3-642-35289-8)  
[Coursera Stanford Machine Learning](https://www.coursera.org/course/ml)  
Additional Contribution by Reuben Brasher, [newexo](https://github.com/newexo)

[justin wagle]:https://github.com/horixon
[reuben brasher]:https://github.com/newexo
