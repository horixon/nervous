Nervous
=========

Cross platform Neuralnet

##Tools
  - Neuralnet - forward and backward propagation
  - Logistic - sigmoid

##Platforms
  - Core neuralnet in C/C++
  - iOS swift wrapper

###Neural Net
Neuralnet with forward and backward propagation.  Implemented iOS using [Accelerate Framework](https://developer.apple.com/library/mac/documentation/Accelerate/Reference/AccelerateFWRef/_index.html).  Net training with [steepest decent](http://en.wikipedia.org/wiki/Gradient_descent).  Theta is the bias and weights.  Matrices are expected in [column major](http://en.wikipedia.org/wiki/Row-major_order#Column-major_order) order, columns are continuous in memory.

Example of usage [here](https://github.com/horixon/nervous/blob/master/nervous-ios/Nervous/NervousTests/NervousTests.swift).  The NN was originally written in Objc, and the C/C++ code is still tested in xcode.

###iOS Use
```sh
$ cd nervous
$ ./configure
$ make
$ ./build-ios.sh
```
You need to inlcude the [module maps](http://clang.llvm.org/docs/Modules.html):  
  - xcode add Search Paths - 'Header Search Paths': nervous/lib/include
  - xcode add Search Paths - 'Library Search Paths': nervous/lib  
  - xcode add Swift Compiler - Search Paths, 'Import Paths': nervous/lib  
  
If you dont want to init the cblass stuff there is a swift wrapper framework.
  - Add the above paths to your project
  - Also add in xcode Swift Compiler - Search Paths, 'Import Paths': Nervous 
  - xcode add Search Paths - 'Header Search Paths': Nervous
  - Include the Nervous project
 
#####References
[Pattern Classification](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0471056693.html)  
[Neural Networks:Tricks of the Trade](http://rd.springer.com/book/10.1007/978-3-642-35289-8)  
[Coursera Stanford Machine Learning](https://www.coursera.org/course/ml)  
Additional Contribution by Reuben Brasher, [newexo](https://github.com/newexo)

[justin wagle]:https://github.com/horixon
[reuben brasher]:https://github.com/newexo