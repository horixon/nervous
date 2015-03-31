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
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testExample() {
        let a = testfoo()
        
        XCTAssertEqual(a[0],Float(1))
        XCTAssertEqual(a[1],Float(1))
    }
    
}
