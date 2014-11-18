//
//  Neuralnet.h
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


#import <Foundation/Foundation.h>
#import "XDTraining.h"

@interface XDNeuralnet : NSObject
@property(assign,nonatomic,readonly)int memorySizeThetas;
@property(assign,nonatomic,readonly)int memorySizeActivations;
@property(assign,nonatomic,readonly)int memorySizeGradient;
@property(assign,nonatomic,readonly)int parametersCount;
@property(assign,nonatomic,readonly) int depth;


-(id)initNetWithUnitCounts:(int*)netUnitCounts netDepth:(int)netDepth;

-(int)memorySizeThetas;
-(int)memorySizeActivations;
-(int)memorySizeGradient;

-(float*)netActivationUnits:(float*)activations;
-(void)forwardPropagationOfThetas:(float*)thetas x:(float*)x activations:(float*)activations;
-(void)backPropagationWithThetas:(float*)thetas activations:(float*)activations labled:(float*)y derivative:(float*)derivative differenciator:(id<XDDifferenciator>)differenciator;
@end
