//
//  LayerTools.m
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

@import Accelerate;

#import "Layer.h"
#import "Logistic.h"

@implementation Layer

+(void)columnMajorThetaFromBias:(float*)bias weights:(float*)weights inputUnits:(int)columns outputUnits:(int)rows result:(float*)theta
{
    memcpy(theta, bias, sizeof(float) * rows);
    memcpy(theta + rows , weights, sizeof(float) * (columns*rows));
}

#pragma mark - Feedforward Propagation
+(void)activationOfInput:(float*)x bias:(float*)bias weights:(float*)weights inputUnits:(int)columns outputUnits:(int)rows result:(float*)a1
{
    cblas_sgemv(CblasColMajor,CblasNoTrans,rows,columns,1.0, weights,rows,x,1,0.0,a1,1);
    
    for(int j=0;j<rows;j++)
    {
        a1[j] = [Logistic sigmoidOfFloat:a1[j] + bias[j]];
    }
}


#pragma mark - Back Propagation

//Bias Derivative
+(void)sensitivityOfOutputActivation:(float*)a2 labledData:(float*)y numberOfLabel:(int)labels result:(float*)d2
{
    memcpy(d2, a2, sizeof(float) * labels );
    cblas_saxpy(labels, -1.0,y,1,d2,1);
}

+(void)sensitivityOfInputActivation:(float*)a1  outputActivationSensitivity:(float*)d2 weights:(float*)w inputUnits:(float)columns outputUnits:(float)rows   result:(float*)d1
{
    float size = columns * sizeof(float);
    
    float* d2w1Transpose = (float*)alloca(size);
    float* oneMinusA1 = (float*)alloca(size);
    float* intermediataryResult =(float*)alloca(size);
    
    catlas_sset(columns, 1.0, oneMinusA1, 1);
    
    cblas_sgemv(CblasColMajor,CblasTrans,rows,columns,1.0,w,rows,d2,1,0.0,d2w1Transpose,1);
    
    vDSP_vmul(a1,1,d2w1Transpose,1,intermediataryResult,1,columns);
    
    cblas_saxpy(columns, -1.0,a1,1,oneMinusA1,1);
    
    vDSP_vmul(oneMinusA1,1,intermediataryResult,1,d1,1,columns);
}

//Weight Derivative
+(void)gradientOfOutputSensitivity:(float*)d2 inputActivation:(float*)a1 inputUnits:(int)columns outputUnits:(int)rows result:(float*)gradient
{
    catlas_sset((rows*columns), 0.0, gradient, 1);
    cblas_sger(CblasColMajor,rows,columns,1,d2,1,a1,1,gradient,rows);
}

@end
