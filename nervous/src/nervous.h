//
//  nervous.h
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

#ifdef __cplusplus
#define NERVOUSDECL extern "C"
#else
#define NERVOUSDECL
#endif

#include "netarch.h"

NERVOUSDECL NetArch netarchitecture(const int* unitcounts,int depth);
NERVOUSDECL int memorysizethetas(NetArch net);
NERVOUSDECL int memorysizeactivations(NetArch net);
NERVOUSDECL int memorysizegradient(NetArch net);
NERVOUSDECL int outputactivationsindex(NetArch netarch);
NERVOUSDECL const float* outputactivations(NetArch netarch, const float *activations);
NERVOUSDECL void seed(long seed);
NERVOUSDECL void randomizethetas(NetArch netarch, float *thetas);
NERVOUSDECL void zerobias(NetArch netarch, float *thetas);
NERVOUSDECL void columnmajortheta(const float* bias, const float* weights, int columns, int rows, float* result);
NERVOUSDECL void activation(const float* x, const float *bias, const float *weights, int columns, int rows, float  *a1);
NERVOUSDECL void forwardprop(NetArch netarch, const float* thetas, const float* x ,float* activations);
NERVOUSDECL void vanillabackprop(NetArch netarch, const float *thetas, const float *activations, const float *y, float *derivative);
NERVOUSDECL void steepestdecent(NetArch netarch, const float *grad, float alpha, float *x);
NERVOUSDECL void regularization(NetArch netarch, float *gradient, const float *thetas, float l1RegFactor, float l2RegFactor);
NERVOUSDECL void signtheta(float* x, int count);
NERVOUSDECL float adaptiverate(const float *gradient, float memory, float alpha, float beta, float learningRate, int parametersCounts, float *runningAverageGradient);
NERVOUSDECL int thetacount(int n, int m);