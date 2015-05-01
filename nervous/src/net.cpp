//
//  net.cpp
//  Neuralnet
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
#include <cstring>

#include "nervous.h"
#include "nettools.h"
#include "training.h"
#include "net.h"

extern "C" void forwardprop(NetArch netarch, const float* thetas, const float* x,float* activations) 
{
    memcpy(activations, x, memorysizeactivations(netarch));

    auto *a0 = activations;
    auto *a1 = a0 + netarch.inputunits;

    enumaratethetas(thetas, &netarch, [&a0, &a1](const float *theta, int n, int m, int units) {      
        activation(a0, theta, theta + m, n, m, a1);    
        a0 = a1;
        a1 += m;
    });
}

extern "C" void vanillabackprop(NetArch netarch, const float *thetas, const float *activations, const float *y, float *derivative)
{
	vanillagrad v;
	backprop(netarch, thetas, activations, y, derivative, v);
}
