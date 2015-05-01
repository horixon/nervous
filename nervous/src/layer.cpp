//
//  layer.cpp
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

#include "blasfuncs.h"
#include "mathutil.h"

extern "C" void activation(const float* x, const float *bias, 
	const float *weights, int columns, int rows, float *a1)
{
	Blas.gemv(CblasColMajor, CblasNoTrans, rows, columns, 1.0, weights, rows,
     x, 1, 0.0, a1, 1);
    
    for(int j = 0; j < rows; j++)
    {
        a1[j] = sigmoid(a1[j] + bias[j]);
    }
}

extern "C" void columnmajortheta(const float* bias, const float* weights, 
	int columns, int rows, float* result)
{
    memcpy(result, bias, sizeof(float) * rows);
    memcpy(result + rows , weights, sizeof(float) * (columns*rows));
}