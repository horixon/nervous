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
        blasinit()
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testParametersize() {
        let netarch = Fixture.makenetarch()
        let expected = Int32(sizeof(Float)*64)
        let actual = memorysizethetas(netarch)
        
        XCTAssertEqual(expected, actual)
    }
    
    func testActivationsize() {
        let netarch = Fixture.makenetarch()
        let expected = Int32(sizeof(Float)*16)
        let actual = memorysizeactivations(netarch)
        
        XCTAssertEqual(expected, actual)
    }
    
    func testGradientsize() {
        let netarch = Fixture.makenetarch()
        let expected = Int32(sizeof(Float)*64)
        let actual = memorysizegradient(netarch)
        
        XCTAssertEqual(expected, actual)
    }
    
    func testRandomNetInit()
    {
        let arch = Fixture.makenetarch()
        var thetas = [Float](count: Int(arch.parameterscount), repeatedValue: 0.0)
        seed(42)
        randomizethetas(arch, &thetas)
        println(thetas)
        
        XCTAssertEqualWithAccuracy(thetas[0], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[1], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[2], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[3], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[4], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[5], -0.499671,1e-6);
        XCTAssertEqualWithAccuracy(thetas[6], 0.0245871,1e-6);
        XCTAssertEqualWithAccuracy(thetas[7], 0.235424,1e-6);
        XCTAssertEqualWithAccuracy(thetas[8], -0.236694,1e-6);
        XCTAssertEqualWithAccuracy(thetas[9], -0.123776,1e-6);
        XCTAssertEqualWithAccuracy(thetas[10], -0.303714,1e-6);
        XCTAssertEqualWithAccuracy(thetas[11], 0.475874,1e-6);
        XCTAssertEqualWithAccuracy(thetas[12], 0.0123181,1e-6);
        XCTAssertEqualWithAccuracy(thetas[13], 0.030449,1e-6);
        XCTAssertEqualWithAccuracy(thetas[14], -0.242898,1e-6);
        XCTAssertEqualWithAccuracy(thetas[15], -0.392913,1e-6);
        XCTAssertEqualWithAccuracy(thetas[16], 0.315488,1e-6);
        XCTAssertEqualWithAccuracy(thetas[17], 0.400545,1e-6);
        XCTAssertEqualWithAccuracy(thetas[18], -0.0479714,1e-6);
        XCTAssertEqualWithAccuracy(thetas[19], -0.254611,1e-6);
        XCTAssertEqualWithAccuracy(thetas[20], -0.252592,1e-6);
        XCTAssertEqualWithAccuracy(thetas[21], -0.311726,1e-6);
        XCTAssertEqualWithAccuracy(thetas[22], -0.176676,1e-6);
        XCTAssertEqualWithAccuracy(thetas[23], -0.396543,1e-6);
        XCTAssertEqualWithAccuracy(thetas[24], 0.307374,1e-6);
        XCTAssertEqualWithAccuracy(thetas[25], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[26], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[27], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[28], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[29], 0.0307781,1e-6);
        XCTAssertEqualWithAccuracy(thetas[30], 0.308089,1e-6);
        XCTAssertEqualWithAccuracy(thetas[31], 0.211996,1e-6);
        XCTAssertEqualWithAccuracy(thetas[32], -0.377151,1e-6);
        XCTAssertEqualWithAccuracy(thetas[33], 0.0348911,1e-6);
        XCTAssertEqualWithAccuracy(thetas[34], -0.330036,1e-6);
        XCTAssertEqualWithAccuracy(thetas[35], 0.324253,1e-6);
        XCTAssertEqualWithAccuracy(thetas[36], -0.0316659,1e-6);
        XCTAssertEqualWithAccuracy(thetas[37], -0.0250681,1e-6);
        XCTAssertEqualWithAccuracy(thetas[38], -0.0447169,1e-6);
        XCTAssertEqualWithAccuracy(thetas[39], -0.238387,1e-6);
        XCTAssertEqualWithAccuracy(thetas[40], -0.436196,1e-6);
        XCTAssertEqualWithAccuracy(thetas[41], -0.426619,1e-6);
        XCTAssertEqualWithAccuracy(thetas[42], 0.441408,1e-6);
        XCTAssertEqualWithAccuracy(thetas[43], 0.37158,1e-6);
        XCTAssertEqualWithAccuracy(thetas[44], 0.255661,1e-6);
        XCTAssertEqualWithAccuracy(thetas[45], 0.0633432,1e-6);
        XCTAssertEqualWithAccuracy(thetas[46], 0.240816,1e-6);
        XCTAssertEqualWithAccuracy(thetas[47], 0.110525,1e-6);
        XCTAssertEqualWithAccuracy(thetas[48], -0.13933,1e-6);
        XCTAssertEqualWithAccuracy(thetas[49], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[50], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[51], 0.0,1e-6);
        XCTAssertEqualWithAccuracy(thetas[52], -0.124367,1e-6);
        XCTAssertEqualWithAccuracy(thetas[53], -0.232888,1e-6);
        XCTAssertEqualWithAccuracy(thetas[54], -0.148263,1e-6);
        XCTAssertEqualWithAccuracy(thetas[55], 0.140945,1e-6);
        XCTAssertEqualWithAccuracy(thetas[56], -0.13622,1e-6);
        XCTAssertEqualWithAccuracy(thetas[57], -0.447193,1e-6);
        XCTAssertEqualWithAccuracy(thetas[58], 0.0286582,1e-6);
        XCTAssertEqualWithAccuracy(thetas[59], -0.342301,1e-6);
        XCTAssertEqualWithAccuracy(thetas[60], -0.0461814,1e-6);
        XCTAssertEqualWithAccuracy(thetas[61], -0.170721,1e-6);
        XCTAssertEqualWithAccuracy(thetas[62], -0.307219,1e-6);
        XCTAssertEqualWithAccuracy(thetas[63], -0.435266,1e-6);    
    }

}
