//
//  training.cpp
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

#include "training.h"
#include "nervous.h"
#include "blasfuncs.h"
#include "mathutil.h"
#include "nettools.h"

extern "C" void steepestdecent(NetArch netarch, const float *b, float alpha, float *x)
{
    Blas.axpy(netarch.parameterscount, -alpha, b, 1, x, 1);
}

extern "C" void regularization(NetArch netarch, float *gradient, const float *thetas, float l1RegFactor, float l2RegFactor)
{
    auto *l2g = new float[netarch.parameterscount];
    auto *l1g =  new float[netarch.parameterscount];
    
    auto gradientMemorySize = memorysizegradient(netarch);

    memcpy(l2g, thetas, gradientMemorySize);
    
    zerobias(netarch, l2g);
    
    memcpy(l1g, l2g, gradientMemorySize);
    
    signtheta(l1g, netarch.parameterscount);
    
    Blas.axpy(netarch.parameterscount, l1RegFactor, l1g, 1, gradient, 1);
    Blas.axpy(netarch.parameterscount, l2RegFactor, l2g, 1, gradient, 1);
    
    delete[] l2g;
    delete[] l1g;
}

void outputsensitivity(const float *a2, const float *y, int labels, float *d2)
{
    memcpy(d2, a2, sizeof(float) * labels);
    Blas.axpy(labels, -1.0, y, 1, d2, 1);
}

void inputsensitivity(const float *a1, const float *d2, const float *w, int columns, int rows, float *d1)
{
    auto d2w1Transpose = new float[columns];
    auto oneMinusA1 = new float[columns];
    auto intermediataryResult = new float[columns];
    
    Blas.set(columns, 1.0, oneMinusA1, 1);
    
    Blas.gemv(CblasColMajor,CblasTrans,rows,columns,1.0,w,rows,d2,1,0.0,d2w1Transpose,1);
    
    Blas.vmul(a1, 1, d2w1Transpose, 1, intermediataryResult, 1, columns);
    
    Blas.axpy(columns, -1.0,a1,1,oneMinusA1,1);
    
    Blas.vmul(oneMinusA1,1,intermediataryResult,1,d1,1,columns);
 
    delete[] d2w1Transpose;
    delete[] oneMinusA1;
    delete[] intermediataryResult;
}

void gradoutputsensitivity(const float *d2, const float *a1, int columns, int rows, float *gradient)
{
    Blas.set(rows * columns , 0.0, gradient, 1);
    Blas.ger(CblasColMajor, rows, columns, 1, d2, 1, a1, 1, gradient, rows);
}

void vanillagrad::outputgrad(const float *a1, const float *a2, const float *y, unsigned n, unsigned m, float *delta, float *d2)
{
    outputsensitivity(a2, y, m, delta);
    gradoutputsensitivity(delta, a1, n, m, d2);
}

void vanillagrad::hiddengrad(const float *a1, const float *a0, const float *d2, const float *w, int n, int m, int p, float *delta, float *d1)
{
    inputsensitivity(a1, d2, w, n, m, delta);
    gradoutputsensitivity(delta, a0, p, n, d1);
}

extern "C" float adaptiverate(const float *gradient, float memory, float alpha, float beta, float learningRate, int parametersCounts, float *runningAverageGradient)
{
    Blas.axpy(parametersCounts, -memory, runningAverageGradient,1,runningAverageGradient,1);
    Blas.axpy(parametersCounts, memory, gradient,1,runningAverageGradient,1);
    
    auto euclidianNorm = Blas.nrm2(parametersCounts,runningAverageGradient, 1);

    return learningRate + (learningRate * alpha) * (beta * euclidianNorm - learningRate);
}
