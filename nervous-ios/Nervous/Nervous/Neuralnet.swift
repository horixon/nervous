//
//  Neuralnet.swift
//  Nervous
//
//  data-tools ver. 01
//
//  Copyright (c) Microsoft Corporation
//
//  All rights reserved.
//
//  MIT License
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the ""Software""), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

import Nervous
import Initblas

public func blasinit() {
    initblas()
}

public func testfoo2() -> [Float] {
    var a: [Float] = [3.0, 4.0, 5.0, 7.0, 9.0, 8.0]
    var v: [Float] = [1.0, -1.0]
    var actual: [Float] = [0.0, 0.0, 0.0]
    
    monkey(3, 2, Float(1.0), &a, 3, &v, 1, Float(1.0), &actual, 1)
    return actual
}

public func testfoo() -> [Float] {
    
    var x:[Float] = [0,0]
    
    foo(2, 1, &x)
    
    return x
}

