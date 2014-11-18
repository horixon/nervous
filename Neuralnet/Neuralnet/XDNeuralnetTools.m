//
//  NeuralnetTools.m
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


#import "XDNeuralnetTools.h"
#include <Accelerate/Accelerate.h>
#import "XDMath.h"

@implementation XDNeuralnetTools
+(int)countOfThetaFromInputUnits:(int)n outputUnits:(int)m
{
    return n*m + m;
}

+(float)randomWeightDistributedByInputUnitCount:(int)inputUnitCount
{
    float w = 1/sqrt(inputUnitCount);
    return w * (((arc4random()/(double)UINT32_MAX) - 0.5) * 2);
}

+(void)configRandomTheta:(float*)theta inputUnits:(int)columns outputUnits:(int)rows
{
    catlas_sset(rows, 0.0, theta, 1);
    [XDNeuralnetTools fillWithRandomWeights:theta+rows inputUnits:columns  outputUnits:rows];
}

+(void)fillWithRandomWeights:(float*)weights inputUnits:(int)columns outputUnits:(int)rows
{
    int matrixCount = rows * columns;
    for(int i=0;i<matrixCount;i++)
    {
        weights[i] = [XDNeuralnetTools randomWeightDistributedByInputUnitCount:columns];
    }
}

#pragma mark - Public API
+(void)fillWithRandomWeights:(float*)weights inputUnitCount:(int)inputUnitCount weightsCount:(int)weightCount
{
    for(int i=0;i<weightCount;i++)
    {
        weights[i] = [XDNeuralnetTools randomWeightDistributedByInputUnitCount:inputUnitCount];
    }
}

+(void)enumarateThetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(float)netDepth block:(ThetaBlock)b
{
    float *t = thetas;
    
    for(int i=0;i<netDepth-1;i++)
    {
        int n = netUnitCounts[i];
        int m = netUnitCounts[i+1];
        int thetaUnits = [XDNeuralnetTools countOfThetaFromInputUnits:n outputUnits:m];
        
        b(t,n,m,thetaUnits);
        
        t += thetaUnits;
    }
}

+(void)randomizeThetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth
{
    [XDNeuralnetTools enumarateThetas:thetas netUnitCounts:netUnitCounts netDepth:netDepth block:^(float *theta, int n, int m, int units) {
        [XDNeuralnetTools configRandomTheta:theta inputUnits:n outputUnits:m];
    }];
}

+(void)zeroBiasInThetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth
{
    [XDNeuralnetTools enumarateThetas:thetas netUnitCounts:netUnitCounts netDepth:netDepth block:^(float *theta, int n, int m, int units) {
        catlas_sset(m, 0.0, theta, 1);
    }];
}

+(void)signOfThetas:(float*)thetas paramterCount:(int)paramterCount
{
    [XDMath signOfFloatArray:thetas count:paramterCount];
}

+(void)combineHiddenLayerUnits:(int*)hiddenLayerUnits numberOfHiddenLayer:(int)numberOfHidden withInputUnitsCount:(int)inputUnitsCount outputUnitsCount:(int)outputUnitsCount result:(int*)netUnitCounts
{
    memcpy(netUnitCounts, &inputUnitsCount, sizeof(int));
    memcpy(netUnitCounts + 1, hiddenLayerUnits, sizeof(int) * numberOfHidden );
    memcpy(netUnitCounts + 1 + numberOfHidden, &outputUnitsCount, sizeof(int));
}

@end