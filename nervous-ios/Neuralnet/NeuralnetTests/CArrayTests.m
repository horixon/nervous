//
//  CArrayTests.m
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


#import <XCTest/XCTest.h>
#import "CArray.h"

@interface CArrayTests : XCTestCase

@end

@implementation CArrayTests

- (void)setUp
{
    [super setUp];
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown
{
    // Put teardown code here. This method is called after the invocation of each test method in the class.
    [super tearDown];
}

- (void)testExample
{
    float arr[23] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    float insert[2] = {100,101};
    
    [CArray insertFloats:insert inTarget:arr + 10 insertCount:2 targetCount:23-10];

    
    XCTAssertEqualWithAccuracy(arr[0], 0.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[1], 1.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[2], 2.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[3], 3.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[4], 4.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[5], 5.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[6], 6.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[7], 7.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[8], 8.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[9], 9.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[10], 100.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[11], 101.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[12], 10.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[13], 11.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[14], 12.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[15], 13.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[16], 14.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[17], 15.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[18], 16.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[19], 17.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[20], 18.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[21], 19.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr[22], 20.000000, 1E-6,@"Score wasn't close enough");
    
    

    float arr1[23] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    
    [CArray insertFloats:insert inTarget:arr1 insertCount:2 targetCount:23];
    
    XCTAssertEqualWithAccuracy(arr1[0], 100.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[1], 101.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[2], 0.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[3], 1.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[4], 2.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[5], 3.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[6], 4.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[7], 5.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[8], 6.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[9], 7.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[10], 8.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[11], 9.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[12], 10.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[13], 11.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[14], 12.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[15], 13.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[16], 14.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[17], 15.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[18], 16.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[19], 17.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[20], 18.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[21], 19.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr1[22], 20.000000, 1E-6,@"Score wasn't close enough");
    
    float arr2[23] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20};
    
    [CArray insertFloats:insert inTarget:arr2 + 23 - 2 insertCount:2 targetCount:23];

    XCTAssertEqualWithAccuracy(arr2[0], 0.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[1], 1.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[2], 2.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[3], 3.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[4], 4.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[5], 5.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[6], 6.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[7], 7.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[8], 8.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[9], 9.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[10], 10.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[11], 11.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[12], 12.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[13], 13.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[14], 14.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[15], 15.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[16], 16.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[17], 17.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[18], 18.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[19], 19.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[20], 20.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[21], 100.000000, 1E-6,@"Score wasn't close enough");
    XCTAssertEqualWithAccuracy(arr2[22], 101.000000, 1E-6,@"Score wasn't close enough");
    
    
}

@end
