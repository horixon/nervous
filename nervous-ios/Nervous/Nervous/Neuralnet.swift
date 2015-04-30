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

public class SteepestDescentNet {
    let arch: NetArch
    public var theta: [Float]
    var a: [Float]
    var grad: [Float]
    let alpha: Float
    var counts: [Int32]
    
    public init(netunitcounts: [Int], initialthetas: [Float], alpha: Float) {
        initblas()

        theta = initialthetas
        
        counts = netunitcounts.map {(x: Int) -> Int32 in Int32(x)}
        arch = netarchitecture(counts, Int32(counts.count))
        
        a = [Float](count: Int(arch.units), repeatedValue: 0.0)
        
        grad = [Float](count: Int(arch.parameterscount), repeatedValue: 0.0)

        self.alpha = alpha
    }
    
    public func setactivations(var x: [Float]) {
        forwardprop(arch, theta, x, &a)
    }
    
    public func updatetheta(var observations: [Float], var x: [Float]) {
        setactivations(x)
        vanillabackprop(arch, theta, &a, observations, &grad)
        steepestdecent(arch, grad, alpha, &theta)
    }
    
    public func activations() -> [Float]{
        return Array(a[a.count - Int(arch.outputunits) ... a.count - 1])
    }
}

public func layersize(layernumber:Int, netunitcounts:[Int]) -> Int {
    let counts = netunitcounts.map {(x: Int) -> Int32 in Int32(x)}
    let inputunits = Int32(counts[layernumber])
    let output = Int32(counts[layernumber + 1])
    
    return Int(thetacount(inputunits,output))
}