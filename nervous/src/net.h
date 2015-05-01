//
//  net.h
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
#pragma once
#include "netarch.h"
#include "nettools.h"

template<typename T, typename DIFFERENTIATOR>
void backprop(const NetArch &netarch, const T *thetas, const T *activations, const T *y, T *derivative,
	DIFFERENTIATOR &differentiator)
{
    auto n = netarch.unitcounts[netarch.indexlayerlasthidden];
    auto m = netarch.unitcounts[netarch.indexlayeroutput];
    
    auto outputthetaindex = netarch.parameterscount - thetacount(n, m);
    
    auto a2 = outputactivations(netarch, activations);
    auto a0 = a2 - n;
    auto a1 = a0;
    
    auto biasgrad = derivative + outputthetaindex;
    auto weightgrad = biasgrad;
    
    auto weights = thetas + outputthetaindex + m;
    
    differentiator.outputgrad(a1, a2, y, n, m, biasgrad, weightgrad + m);
    
    for(auto i = netarch.indexlayeroutput; i > 1; i--)
    {
        n = netarch.unitcounts[i - 1];
        m = netarch.unitcounts[i];
        
        auto p = netarch.unitcounts[i - 2];
        
        T *layerbiasgrad = biasgrad;
        biasgrad -= thetacount(p, n);
        weightgrad = biasgrad + n;
        
        a0 -=p;
        
        differentiator.hiddengrad(a1, a0, layerbiasgrad, weights, n, m, p, biasgrad, weightgrad);
        
        a1 = a0;
        weights -= n * p + m;
    }
}
