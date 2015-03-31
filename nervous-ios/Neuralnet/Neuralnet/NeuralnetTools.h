//
//  NeuralnetTools.h
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


@import Foundation;

typedef void(^ThetaBlock)(float* theta, int n, int m, int units);

@interface NeuralnetTools : NSObject
+(void)enumarateThetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(float)netDepth block:(ThetaBlock)b;
+(void)randomizeThetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth;
+(void)zeroBiasInThetas:(float*)thetas netUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth;
+(void)signOfThetas:(float*)thetas paramterCount:(int)paramterCount;
+(int)countOfThetaFromInputUnits:(int)n outputUnits:(int)m;
+(void)fillWithRandomWeights:(float*)weights inputUnitCount:(int)inputUnitCount weightsCount:(int)weightCount;
+(void)combineHiddenLayerUnits:(int*)hiddenLayerUnits numberOfHiddenLayer:(int)numberOfHidden withInputUnitsCount:(int)inputUnitsCount outputUnitsCount:(int)outputUnitsCount result:(int*)netUnitCounts;
+(float)randomWeightDistributedByInputUnitCount:(int)inputUnitCount;
@end
