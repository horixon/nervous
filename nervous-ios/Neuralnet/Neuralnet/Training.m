//
//  Learning.m
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

#import "Training.h"
#import "Layer.h"
#import "NeuralnetTools.h"

@implementation Training
+(float)adaptiveLearningWithSteepestDecentOfGradient:(float*)gradient memory:(float)memory alpha:(float)alpha beta:(float)beta learningRate:(float)learningRate parametersCount:(int)parametersCounts updateRunningGradient:(float*)runningAverageGradient
{
    cblas_saxpy(parametersCounts, -memory, runningAverageGradient,1,runningAverageGradient,1);
    cblas_saxpy(parametersCounts, memory, gradient,1,runningAverageGradient,1);
    
    float euclidianNorm = cblas_snrm2(parametersCounts,runningAverageGradient,1);

    return learningRate + (learningRate * alpha)*(beta*euclidianNorm - learningRate);
}

@end

@implementation Derivatives
+(void)steepestDecentUpdateGradient:(float*)b alpha:(float)learningRate parameterCount:(int)ps x:(float*)x
{
    cblas_saxpy(ps, -learningRate, b, 1, x,1);
}

+(void)regularizationOfGradient:(float*)gradient gradientMemorySize:(int)gradientMemorySize thetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth parameterCount:(int)parameterCount l1:(float)l1RegFactor l2:(float)l2RegFactor
{
    //float* l2g = malloc(gradientMemorySize);
    //float* l1g = malloc(gradientMemorySize);
    
    float* l2g = alloca(gradientMemorySize);
    float* l1g = alloca(gradientMemorySize);
    
    memcpy(l2g,thetas,gradientMemorySize);
    
    [NeuralnetTools zeroBiasInThetas:l2g netUnitCounts:netUnitCounts netDepth:netDepth];
    
    memcpy(l1g,l2g,gradientMemorySize);
    
    [NeuralnetTools signOfThetas:l1g paramterCount:parameterCount];
    
    cblas_saxpy(parameterCount, l1RegFactor, l1g,1,gradient,1);
    cblas_saxpy(parameterCount, l2RegFactor, l2g,1,gradient,1);
    
    //free(l2g);
    //free(l1g);
}
@end

@implementation SteepestDecentTraining
-(void)outputLayerDerivativeFromInputActivation:(float*)a1 outputActivation:(float*)a2 labledData:(float*)y inputUnits:(int)n outputUnits:(int)m  biasGradient:(float*)delta weightGradient:(float*)d2
{
    [Layer sensitivityOfOutputActivation:a2 labledData:y numberOfLabel:m result:delta];
    [Layer gradientOfOutputSensitivity:delta inputActivation:a1 inputUnits:n outputUnits:m  result:d2];
}
-(void)hiddenLayerDerivativeFromInputActivation:(float*)a1 upperLayerActivation:(float*)a0 lowerLayerBiasGradient:(float*)d2 weights:(float*)w inputUnits:(int)n outputUnits:(int)m upperLayerUnits:(int)p  biasGradient:(float*)delta weightGradient:(float*)d1
{
    [Layer sensitivityOfInputActivation:a1 outputActivationSensitivity:d2 weights:w inputUnits:n outputUnits:m result:delta];
    [Layer gradientOfOutputSensitivity:delta inputActivation:a0 inputUnits:p outputUnits:n result:d1];
}
@end