//
//  Neuralnet.m
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


#import "Net.h"
#import "Layer.h"
#import "NeuralnetTools.h"
#import "Training.h"

@interface Net()
@property(assign,nonatomic,readonly) int* netUnitCounts;
@property(assign,nonatomic,readonly) int indexOfOutputLayer;
@property(assign,nonatomic,readonly) int indexOfLastHiddenLayer;
@property(assign,nonatomic,readonly) int units;
@property(assign,nonatomic,readonly) int inputUnits;
@property(assign,nonatomic,readonly) int netOutputUnits;
@end

@implementation Net

#pragma mark - Private API
-(void)configNetSize
{
    _indexOfOutputLayer = _depth - 1;
    _indexOfLastHiddenLayer = _depth - 2;
    _inputUnits = _netUnitCounts[0];
    _netOutputUnits = _netUnitCounts[_indexOfOutputLayer];

    _parametersCount = 0,
    _units = _inputUnits;
    
    for(int i=0;i < _indexOfOutputLayer;i++)
    {
        int n = _netUnitCounts[i];
        int m = _netUnitCounts[i+1];
        
        _parametersCount += [NeuralnetTools countOfThetaFromInputUnits:n outputUnits:m];
        _units += m;
    }
}

#pragma mark - Public API
-(int)memorySizeThetas
{
    return sizeof(float)*_parametersCount;
}
-(int)memorySizeActivations
{
    return sizeof(float) * _units;
}
-(int)memorySizeGradient
{
    return [self memorySizeThetas];
}

#pragma mark - Public Net API

-(float*)netActivationUnits:(float*)activations
{
    int netActivationIndex = (_units - _netOutputUnits);

    return activations + netActivationIndex;
}

-(void)forwardPropagationOfThetas:(float*)thetas x:(float*)x activations:(float*)activations
{
    memcpy(activations,x,self.memorySizeActivations);
    
    __block float *a0,*a1,*theta;
    
    a0 = activations,
    a1 = a0 + _inputUnits;
    
    theta = thetas;
    
    [NeuralnetTools enumarateThetas:thetas netUnitCounts:_netUnitCounts netDepth:_depth block:^(float *theta, int n, int m, int units) {
        
        [Layer activationOfInput:a0 bias:theta weights:theta + m inputUnits:n outputUnits:m result:a1];
        
        a0 = a1;
        
        a1 += m;
    }];
}

-(void)backPropagationWithThetas:(float*)thetas activations:(float*)activations labled:(float*)y derivative:(float*)derivative differentiator:(id<Differentiator>)differentiator
{
    float *a0,*a1,*a2,*weights,*biasGradient,*weightGradient;
    
    int n = _netUnitCounts[_indexOfLastHiddenLayer],
    m = _netUnitCounts[_indexOfOutputLayer];
    
    int outputThetaUnitCount = [NeuralnetTools countOfThetaFromInputUnits:n outputUnits:m],
    outputThetaIndex = (_parametersCount - outputThetaUnitCount);
    
    a2 = [self netActivationUnits:activations],
    a0 = a1 = a2 - n;
    
    biasGradient = derivative + outputThetaIndex,
    weightGradient = biasGradient;
    
    weights = (thetas + outputThetaIndex) + m;
    
    [differentiator outputLayerDerivativeFromInputActivation:a1 outputActivation:a2 labledData:y inputUnits:n outputUnits:m  biasGradient:biasGradient weightGradient:weightGradient + m];
    
      for(int i=_indexOfOutputLayer;i > 1;i--)
    {
        n = _netUnitCounts[i - 1];
        m = _netUnitCounts[i];
        
        int p = _netUnitCounts[i - 2];
        
        float *lowerLayerBiasGradient = biasGradient;
        biasGradient -= [NeuralnetTools countOfThetaFromInputUnits:p outputUnits:n];
        weightGradient = biasGradient + n;
        
        a0 -=p;
        
        [differentiator hiddenLayerDerivativeFromInputActivation:a1 upperLayerActivation:a0 lowerLayerBiasGradient:lowerLayerBiasGradient weights:weights inputUnits:n outputUnits:m upperLayerUnits:p biasGradient:biasGradient weightGradient:weightGradient];
        
        a1 = a0;
        weights -= n*p + m;
    }
}

-(id)initNetWithUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth
{
    if (self = [super init])
    {
        _depth = netDepth;
        _netUnitCounts = netUnitCounts;
        [self configNetSize];
    }
    return self;
}
@end