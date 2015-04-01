//
//  NervousTests.swift
//  NervousTests
//
//  Created by justin on 3/31/15.
//  Copyright (c) 2015 Microsoft. All rights reserved.
//

import UIKit
import XCTest
import Nervous

class NervousTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        blasinit()
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testFoo() {
        let actual = testfoo()
        let expected: [Float] = [1, 1]
        XCTAssertEqual(expected, actual)
    }
    
    func testMonkey() {
        let actual = testfoo2()
        let expected = [-4, -5, -3]
        XCTAssertEqual(expected, actual)
    }
    
}
