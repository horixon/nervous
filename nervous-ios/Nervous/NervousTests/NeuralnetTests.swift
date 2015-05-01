//
//  
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

import Nervous
import Nerves
import XCTest

class NeuralnetTests: XCTestCase {

    override func setUp() {
        super.setUp()
        // Put setup code here. This method is called before the invocation of each test method in the class.
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }

    func testActivations() {
        var nn = NetFixture.getNet()
        nn.setactivations(NetFixture.x)
        let expected =  NetFixture.a
        let actual = nn.activations()
        
        XCTAssertEqual(expected, actual)
    }
    
    func testUpdateNet() {
        let epsilon = 1e-6
        var nn = NetFixture.getNet()
        let y: [Float] = [1.0, 0.0, 0.0]
        nn.updatetheta(y, x: NetFixture.x)
        let thetas = nn.theta.map{Double($0)}
        
        XCTAssertEqualWithAccuracy(thetas[0], 0.271825, epsilon)
        XCTAssertEqualWithAccuracy(thetas[1], -0.403744, epsilon)
        XCTAssertEqualWithAccuracy(thetas[2], -0.439635, epsilon)
        XCTAssertEqualWithAccuracy(thetas[3], -0.386128, epsilon)
        XCTAssertEqualWithAccuracy(thetas[4], -0.157618, epsilon)
        XCTAssertEqualWithAccuracy(thetas[5], -0.588133, epsilon)
        XCTAssertEqualWithAccuracy(thetas[6], 0.356090, epsilon)
        XCTAssertEqualWithAccuracy(thetas[7], -0.400308, epsilon)
        XCTAssertEqualWithAccuracy(thetas[8], 0.031978, epsilon)
        XCTAssertEqualWithAccuracy(thetas[9], -0.561632, epsilon)
        XCTAssertEqualWithAccuracy(thetas[10], -0.613667, epsilon)
        XCTAssertEqualWithAccuracy(thetas[11], 0.008328, epsilon)
        XCTAssertEqualWithAccuracy(thetas[12], 0.004203, epsilon)
        XCTAssertEqualWithAccuracy(thetas[13], -0.754533, epsilon)
        XCTAssertEqualWithAccuracy(thetas[14], -0.661538, epsilon)
        XCTAssertEqualWithAccuracy(thetas[15], 0.036628, epsilon)
        XCTAssertEqualWithAccuracy(thetas[16], -0.536184, epsilon)
        XCTAssertEqualWithAccuracy(thetas[17], -0.294820, epsilon)
        XCTAssertEqualWithAccuracy(thetas[18], 0.514078, epsilon)
        XCTAssertEqualWithAccuracy(thetas[19], 0.337003, epsilon)
        XCTAssertEqualWithAccuracy(thetas[20], -0.541552, epsilon)
        XCTAssertEqualWithAccuracy(thetas[21], 0.425978, epsilon)
        XCTAssertEqualWithAccuracy(thetas[22], -0.194625, epsilon)
        XCTAssertEqualWithAccuracy(thetas[23], 0.126278, epsilon)
        XCTAssertEqualWithAccuracy(thetas[24], 0.516715, epsilon)
        
        XCTAssertEqualWithAccuracy(thetas[25], 0.063360, epsilon)
        XCTAssertEqualWithAccuracy(thetas[26], 0.088831, epsilon)
        XCTAssertEqualWithAccuracy(thetas[27], -0.821839, epsilon)
        XCTAssertEqualWithAccuracy(thetas[28], -0.799743, epsilon)
        XCTAssertEqualWithAccuracy(thetas[29], -0.745080, epsilon)
        XCTAssertEqualWithAccuracy(thetas[30], -0.490228, epsilon)
        XCTAssertEqualWithAccuracy(thetas[31], 0.054165, epsilon)
        XCTAssertEqualWithAccuracy(thetas[32], -0.742476, epsilon)
        XCTAssertEqualWithAccuracy(thetas[33], 0.343966, epsilon)
        XCTAssertEqualWithAccuracy(thetas[34], -0.721178, epsilon)
        XCTAssertEqualWithAccuracy(thetas[35], 0.493126, epsilon)
        XCTAssertEqualWithAccuracy(thetas[36], 0.448265, epsilon)
        XCTAssertEqualWithAccuracy(thetas[37], -0.575817, epsilon)
        XCTAssertEqualWithAccuracy(thetas[38], -0.293299, epsilon)
        XCTAssertEqualWithAccuracy(thetas[39], 0.401781, epsilon)
        XCTAssertEqualWithAccuracy(thetas[40], 0.198074, epsilon)
        XCTAssertEqualWithAccuracy(thetas[41], 0.662777, epsilon)
        XCTAssertEqualWithAccuracy(thetas[42], 0.154145, epsilon)
        XCTAssertEqualWithAccuracy(thetas[43], -0.191981, epsilon)
        XCTAssertEqualWithAccuracy(thetas[44], -0.636180, epsilon)
        XCTAssertEqualWithAccuracy(thetas[45], 0.313975, epsilon)
        XCTAssertEqualWithAccuracy(thetas[46], 0.541250, epsilon)
        XCTAssertEqualWithAccuracy(thetas[47], -0.576212, epsilon)
        XCTAssertEqualWithAccuracy(thetas[48], -0.415489, epsilon)
        
        XCTAssertEqualWithAccuracy(thetas[49], 0.656614, epsilon)
        XCTAssertEqualWithAccuracy(thetas[50], 0.049338, epsilon)
        XCTAssertEqualWithAccuracy(thetas[51], 0.130676, epsilon)
        XCTAssertEqualWithAccuracy(thetas[52], -0.311475, epsilon)
        XCTAssertEqualWithAccuracy(thetas[53], -0.044574, epsilon)
        XCTAssertEqualWithAccuracy(thetas[54], -0.576350, epsilon)
        XCTAssertEqualWithAccuracy(thetas[55], 0.335253, epsilon)
        XCTAssertEqualWithAccuracy(thetas[56], -0.889540, epsilon)
        XCTAssertEqualWithAccuracy(thetas[57], -0.944205, epsilon)
        XCTAssertEqualWithAccuracy(thetas[58], -0.607657, epsilon)
        XCTAssertEqualWithAccuracy(thetas[59], -0.196923, epsilon)
        XCTAssertEqualWithAccuracy(thetas[60], -0.286429, epsilon)
        XCTAssertEqualWithAccuracy(thetas[61], -0.804256, epsilon)
        XCTAssertEqualWithAccuracy(thetas[62], 0.345382, epsilon)
        XCTAssertEqualWithAccuracy(thetas[63], -0.458212, epsilon)
    }

}
