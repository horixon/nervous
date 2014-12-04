Data-Tools
=========

For doing fun things with data on iOS or osx.

##Tools
  - Neuralnet - forward and backward propagation
  - Logistic - sigmoid

###Neural Net
Neuralnet with forward and backward propagation.  Implemented using [Accelerate Framework](https://developer.apple.com/library/mac/documentation/Accelerate/Reference/AccelerateFWRef/_index.html).  Net training with [steepest decent](http://en.wikipedia.org/wiki/Gradient_descent).  Theta is the bias and weights.  Matrices are expected in [column major](http://en.wikipedia.org/wiki/Row-major_order#Column-major_order) order, columns are continuous in memory.

In practice with a mobile use case you may want to first train the model across a large data set not on the device. After the model is trained use it on the device where, if you choose, it will continue to learn.

Example of usage [here](https://github.com/horixon/data-tools/blob/master/Neuralnet/NeuralnetTests/NeuralnetTests.m).  The consumer of Neuralnet.m is responsible for handling the memory allocation.

#####References
[Pattern Classification](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0471056693.html)  
[Neural Networks:Tricks of the Trade](http://rd.springer.com/book/10.1007/978-3-642-35289-8)  
[Coursera Stanford Machine Learning](https://www.coursera.org/course/ml)  
Additional Contribution by Reuben Brasher, [newexo](https://github.com/newexo)


[justin wagle]:https://github.com/horixon
[reuben brasher]:https://github.com/newexo
