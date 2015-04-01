//
//  NetSizeTests.swift
//  Nervous
//
//  Created by justin on 3/31/15.
//  Copyright (c) 2015 Microsoft. All rights reserved.
//

import UIKit
import XCTest
import Nervous

class NetSizeTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func makenetarch()-> NetArch {
        return netarchitecture([4,5,4,3],4)
    }
    
    func testParametersize() {
        let netarch = makenetarch()
        let expected = Int32(sizeof(Float)*64)
        let actual = memorysizethetas(netarch)
        
        XCTAssertEqual(expected, actual)
    }
    
    func testActivationsize() {
        let netarch = makenetarch()
        let expected = Int32(sizeof(Float)*16)
        let actual = memorysizeactivations(netarch)
        
        XCTAssertEqual(expected, actual)
    }
    
    func testGradientsize() {
        let netarch = makenetarch()
        let expected = Int32(sizeof(Float)*64)
        let actual = memorysizegradient(netarch)
        
        XCTAssertEqual(expected, actual)
    }
    
    /*
    -(void)testRandomNetInit
    {
    int netUnitCounts[] = {4, 5, 4, 3};
    Net *net = [[Net alloc] initNetWithUnitCounts:netUnitCounts netDepth:4];
    
    float *thetas = alloca(net.memorySizeThetas);
    [NeuralnetTools randomizeThetas:thetas netUnitCounts:netUnitCounts netDepth:4];
    
    XCTAssertTrue(thetas[0] == 0);
    XCTAssertTrue(thetas[1] == 0);
    XCTAssertTrue(thetas[2] == 0);
    XCTAssertTrue(thetas[3] == 0);
    
    XCTAssertTrue(thetas[25] == 0);
    XCTAssertTrue(thetas[26] == 0);
    XCTAssertTrue(thetas[27] == 0);
    XCTAssertTrue(thetas[28] == 0);
    
    XCTAssertTrue(thetas[49] == 0);
    XCTAssertTrue(thetas[50] == 0);
    XCTAssertTrue(thetas[51] == 0);
    }
    
    -(void)testNetUnitCountsFromHiddens
    {
    int inputUnitCount = 90;
    int outputUnitCount = 91;
    
    int hlUnitLength = 5;
    int hlUnitCounts[5] = {21,22,23,24,25};
    
    int *netUnitCounts = malloc( sizeof(int) * (2 + 5) );
    
    [NeuralnetTools combineHiddenLayerUnits:hlUnitCounts numberOfHiddenLayer:hlUnitLength withInputUnitsCount:inputUnitCount outputUnitsCount:outputUnitCount result:netUnitCounts];
    
    XCTAssertTrue(netUnitCounts[0] == 90);
    XCTAssertTrue(netUnitCounts[1] == 21);
    XCTAssertTrue(netUnitCounts[2] == 22);
    XCTAssertTrue(netUnitCounts[3] == 23);
    XCTAssertTrue(netUnitCounts[4] == 24);
    XCTAssertTrue(netUnitCounts[5] == 25);
    XCTAssertTrue(netUnitCounts[6] == 91);    
    }

    */

}
