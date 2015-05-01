//
//  nettools.cpp
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
#include "netarch.h"
#include "nettools.h"
#include "blasfuncs.h"

#include <cmath>
#include <cstring>
#include <random>

using namespace std;

uniform_real_distribution<float> distribution(-1.0, 1.0);
minstd_rand0 generatortimer;

extern "C" void seed(long seed)
{
    generatortimer.seed(seed);
}

extern "C" int thetacount(int n, int m) 
{
    return n * m + m;
}

float randomweight(int inputUnitCount)
{
    float w = 1/sqrt(inputUnitCount);
    return w * distribution(generatortimer);
}

void randomweight(float *weights, int columns, int rows)
{
    int matrixcount = rows * columns;
    for(int i = 0; i < matrixcount; i++)
    {
        weights[i] = randomweight(columns);
    }
}

void randomtheta(float* theta, int columns, int rows)
{
	Blas.set(rows, 0.0, theta, 1);

    randomweight(theta + rows, columns, rows);
}

extern "C" void randomizethetas(NetArch netarch, float *thetas)
{
	enumaratethetas(thetas, &netarch, [](float *theta, int n, int m, int units) {      
        randomtheta(theta, n, m);
    });
}

extern "C" void zerobias(NetArch netarch, float *thetas)
{
    enumaratethetas(thetas, &netarch, [](float *theta, int n, int m, int units) {      
        Blas.set(m, 0.0, theta, 1);
    });
}
