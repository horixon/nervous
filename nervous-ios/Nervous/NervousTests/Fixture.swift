//
//  Fixture.swift
//  Nervous
//
//  Created by justin on 4/30/15.
//  Copyright (c) 2015 Microsoft. All rights reserved.
//

import Foundation
import Nervous

struct Fixture {
    static let units: [Int32] = [4,5,4,3]
    
    static func makenetarch()-> NetArch {
        return netarchitecture(units, 4)
    }
    
}