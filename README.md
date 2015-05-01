Nervous
=========

Cross platform Neuralnet

##Tools
  - Neuralnet - forward and backward propagation
  - Logistic - sigmoid

##Platforms
  - Core neuralnet in C++ with C API
  - iOS Swift wrapper

###Neural Net
Neuralnet with forward and backward propagation. Theta is the bias and weights.  Matrices are expected in [column major](http://en.wikipedia.org/wiki/Row-major_order#Column-major_order) order, columns are continuous in memory.  You need to set up the [blas function pointers](nervous/src/blasfuncs.h), the ios wrapper uses [Accelerate Framework](https://developer.apple.com/library/mac/documentation/Accelerate/Reference/AccelerateFWRef/_index.html).  Vanilla net training with [steepest decent](http://en.wikipedia.org/wiki/Gradient_descent).  

Example usage [here](nervous-ios/Nervous/NervousTests/NervousTests.swift).  The NN was originally written in Objc, the [tests](nervous-ios/Nervous/NervousTests) haven't been migrated to C++ yet so you need xcode to run the tests.

###Use
```sh
$ cd nervous
$ ./configure
$ make
for ios
$ ./build-ios.sh
```
For ios/osx you need to include inlcude the [module maps](http://clang.llvm.org/docs/Modules.html) and lib in your project and set up the [clbas function pointers](nervous-ios/Nervous/Nervous/initblas.c):  
  - xcode add Search Paths -> 'Header Search Paths': nervous/lib/include
  - xcode add Search Paths -> 'Library Search Paths': nervous/lib  
  - xcode add Swift Compiler -> Search Paths -> 'Import Paths': nervous/lib  
  
There is a Swift wrapper framework with a [basic NN wrapper](nervous-ios/Nervous).
  - Add the paths above to your project
  - xcode add Swift Compiler -> Search Paths -> 'Import Paths': Nervous 
  - xcode add Search Paths -> 'Header Search Paths': Nervous
  - Include Nervous project in your workspace
 
#####References
[Pattern Classification](http://www.wiley.com/WileyCDA/WileyTitle/productCd-0471056693.html)  
[Neural Networks:Tricks of the Trade](http://rd.springer.com/book/10.1007/978-3-642-35289-8)  
[Coursera Stanford Machine Learning](https://www.coursera.org/course/ml)  
Additional Contribution by Reuben Brasher, [newexo](https://github.com/newexo)

[justin wagle]:https://github.com/horixon
[reuben brasher]:https://github.com/newexo
