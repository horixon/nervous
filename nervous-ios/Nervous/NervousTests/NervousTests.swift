//
//  NervousTests.swift
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

import UIKit
import XCTest
import Nervous
import Nerves

class NervousTests: XCTestCase {
    
    override func setUp() {
        super.setUp()
        blasinit()
    }
    
    override func tearDown() {
        // Put teardown code here. This method is called after the invocation of each test method in the class.
        super.tearDown()
    }
    
    func testUnitsActivation() {

        let weights:[Float] = [0.2, -0.3,-0.4, 0.1,-0.1, 0.3]
        let bias:[Float] = [-0.5,0.5]
        let x:[Float] = [0.5, 0.2, 0.8]
        var result = [Float](count: weights.count + bias.count, repeatedValue: 0.0)
        
        columnmajortheta(bias,weights,3,2,&result)
        
        let w = [Float](result[2..<result.count])
        
        activation(x, result, w, Int32(x.count), Int32(bias.count), &result)
    
        XCTAssertEqualWithAccuracy(result[0], 0.36354745971843, 1e-6)
        XCTAssertEqualWithAccuracy(result[1], 0.64794080208065, 1e-6)
    }

    func testLogisticRegression() {

        let features:[Float] = [1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0,1.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
        let weights:[Float] = [-1.07601,-0.6486902,-0.02074373,1.31564,0.1640269,-0.6585848,0.1790922,0.2390806,0.1969987,0.5717129,1.193347,0.7632604,0.6918613,0.05741632,0.06782798,-2.462446,-1.736741,-0.3176472,1.086548,-2.018481,-0.1100008,-0.7384842,1.541484,0.2833921,0.3675288,-0.2966308,-0.8814878,1.090011,1.145712,-0.6256692,0.1761505,0.2613492,-0.201963,0.7829806,-0.235694,-0.4683684,0.004706901,0.6933804,1.989245]
        let bias:[Float] = [0.7690496]
            
        var result:[Float] = [0]
        
        activation(features, bias, weights, Int32(features.count), Int32(1), &result)

        XCTAssertEqualWithAccuracy(result[0], 0.823099792003632, 1e-6);
    }
    
    func testnet() {
        let x:[Float] = [0.645569620253164, 0.795454545454545, 0.202898550724638,0.04]
        
        let t0:[Float] = [0.274516709297235, -0.387684537831344, -0.433415232653988, -0.391518871473416, -0.167429949511106, -0.586395926228306, 0.366457325061409, -0.396292497768153, 0.0284975266838558, -0.567966321615577, -0.611526072562978, 0.0211028302125387, 0.00915055009475814, -0.758821193475008, -0.669343304516762, 0.0371739070838091, -0.532925406900682, -0.293557975832243, 0.512984186608726, 0.33501202556527, -0.541444644635845, 0.426620375828807, -0.194375807203831, 0.126061934649443, 0.516322961794674]
        
        let t1:[Float] = [0.0652716873048915, 0.026803482797526, -0.810038320204754, -0.7734843343635, -0.74440460639501, -0.512141836316264, 0.0583338156896525, -0.73319921779222, 0.344814001173543, -0.748711089671645, 0.498364147182513, 0.459920956746912, -0.575203443502579, -0.313208638993292, 0.405568429614419, 0.206502410954253, 0.663342658728012, 0.135800179315184, -0.188490372146165, -0.628414354101992, 0.314497700533244, 0.524297197694471, -0.572986156487358, -0.408312257707438]
        
        let t2:[Float] = [0.504439236842861, 0.193606173462437, 0.237868443768336, -0.389572731064328, 0.0294658646219818, -0.521337561593108, 0.274312917530031, -0.831766613405672, -0.901278877250215, -0.659838494042704, -0.147453220554047, -0.24967234028206, -0.843425153116835, 0.382516139902567, -0.4306203872368]
        
        let arch = Fixture.makenetarch()
        let thetamemsize = memorysizethetas(arch)
        
        var thetas = t0 + t1 + t2
        
        //Forward Propagation
        var activations = [Float](count: Int(arch.units), repeatedValue: 0.0)
        
        forwardprop(arch, thetas, x, &activations)
        let netActivation = outputactivations(arch, &activations)
        
        XCTAssertEqualWithAccuracy(activations[0], 0.645569620253164, 1E-6)
        XCTAssertEqualWithAccuracy(activations[1], 0.795454545454545, 1E-6)
        XCTAssertEqualWithAccuracy(activations[2], 0.202898550724638, 1E-6)
        XCTAssertEqualWithAccuracy(activations[3], 0.04, 1E-6)
        
        XCTAssertEqualWithAccuracy(activations[4], 0.353292190617222, 1E-6)
        XCTAssertEqualWithAccuracy(activations[5], 0.443889969220041, 1E-6)
        XCTAssertEqualWithAccuracy(activations[6], 0.320972036722546, 1E-6)
        XCTAssertEqualWithAccuracy(activations[7], 0.295753254990652, 1E-6)
        XCTAssertEqualWithAccuracy(activations[8], 0.273309154195735, 1E-6)
        
        XCTAssertEqualWithAccuracy(activations[9], 0.513210789512956, 1E-6)
        XCTAssertEqualWithAccuracy(activations[10], 0.40045966172715, 1E-6)
        XCTAssertEqualWithAccuracy(activations[11], 0.34290312827802, 1E-6)
        XCTAssertEqualWithAccuracy(activations[12], 0.257398678168021, 1E-6)
        
        XCTAssertEqualWithAccuracy(activations[13], 0.492750598745034, 1E-6)
        XCTAssertEqualWithAccuracy(activations[14], 0.48089340194659, 1E-6)
        XCTAssertEqualWithAccuracy(activations[15], 0.35730787433382, 1E-6)
        
        XCTAssertEqualWithAccuracy(netActivation[0], 0.492750598745034, 1E-6)
        XCTAssertEqualWithAccuracy(netActivation[1], 0.48089340194659, 1E-6)
        XCTAssertEqualWithAccuracy(netActivation[2], 0.35730787433382, 1E-6)
        
        let activationslice = activations[Int(outputactivationsindex(arch)) ..< activations.count]
        XCTAssertEqualWithAccuracy(activationslice[0], 0.492750598745034, 1E-6)
        XCTAssertEqualWithAccuracy(activationslice[1], 0.48089340194659, 1E-6)
        XCTAssertEqualWithAccuracy(activationslice[2], 0.35730787433382, 1E-6)
        
        //Backward Propagation
        let y:[Float] = [1.0,0.0,0.0]
        
        var gradient = [Float](count: Int(arch.parameterscount), repeatedValue: 0.0)

        vanillabackprop(arch, thetas, activations, y, &gradient)
        
        XCTAssertEqualWithAccuracy(gradient[0], 0.008972, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[1], 0.053532, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[2], 0.020732, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[3], -0.017969, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[4], -0.032707, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[5], 0.005792, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[6], 0.034559, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[7], 0.013384, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[8], -0.011600, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[9], -0.021115, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[10], 0.007137, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[11], 0.042583, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[12], 0.016491, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[13], -0.014293, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[14], -0.026017, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[15], 0.001820, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[16], 0.010862, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[17], 0.004206, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[18], -0.003646, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[19], -0.006636, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[20], 0.000359, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[21], 0.002141, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[22], 0.000829, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[23], -0.000719, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[24], -0.001308, 1e-6)
        
        XCTAssertEqualWithAccuracy(gradient[25], 0.006371, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[26], -0.206760, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[27], 0.039337, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[28], 0.087527, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[29], 0.002251, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[30], -0.073047, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[31], 0.013898, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[32], 0.030923, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[33], 0.002828, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[34], -0.091779, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[35], 0.017461, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[36], 0.038853, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[37], 0.002045, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[38], -0.066364, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[39], 0.012626, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[40], 0.028094, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[41], 0.001884, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[42], -0.061150, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[43], 0.011634, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[44], 0.025887, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[45], 0.001741, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[46], -0.056509, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[47], 0.010751, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[48], 0.023922, 1e-6)
        
        XCTAssertEqualWithAccuracy(gradient[49], -0.507249, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[50], 0.480893, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[51], 0.357308, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[52], -0.260326, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[53], 0.246800, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[54], 0.183374, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[55], -0.203133, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[56], 0.192578, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[57], 0.143087, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[58], -0.173937, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[59], 0.164900, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[60], 0.122522, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[61], -0.130565, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[62], 0.123781, 1e-6)
        XCTAssertEqualWithAccuracy(gradient[63], 0.091971, 1e-6)

        //Learning
        
        steepestdecent(arch, gradient, 0.3, &thetas)
        
        XCTAssertEqualWithAccuracy(thetas[0], 0.271825, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[1], -0.403744, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[2], -0.439635, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[3], -0.386128, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[4], -0.157618, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[5], -0.588133, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[6], 0.356090, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[7], -0.400308, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[8], 0.031978, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[9], -0.561632, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[10], -0.613667, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[11], 0.008328, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[12], 0.004203, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[13], -0.754533, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[14], -0.661538, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[15], 0.036628, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[16], -0.536184, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[17], -0.294820, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[18], 0.514078, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[19], 0.337003, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[20], -0.541552, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[21], 0.425978, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[22], -0.194625, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[23], 0.126278, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[24], 0.516715, 1e-6)
        
        XCTAssertEqualWithAccuracy(thetas[25], 0.063360, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[26], 0.088831, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[27], -0.821839, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[28], -0.799743, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[29], -0.745080, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[30], -0.490228, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[31], 0.054165, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[32], -0.742476, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[33], 0.343966, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[34], -0.721178, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[35], 0.493126, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[36], 0.448265, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[37], -0.575817, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[38], -0.293299, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[39], 0.401781, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[40], 0.198074, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[41], 0.662777, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[42], 0.154145, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[43], -0.191981, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[44], -0.636180, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[45], 0.313975, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[46], 0.541250, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[47], -0.576212, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[48], -0.415489, 1e-6)
        
        XCTAssertEqualWithAccuracy(thetas[49], 0.656614, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[50], 0.049338, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[51], 0.130676, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[52], -0.311475, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[53], -0.044574, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[54], -0.576350, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[55], 0.335253, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[56], -0.889540, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[57], -0.944205, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[58], -0.607657, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[59], -0.196923, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[60], -0.286429, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[61], -0.804256, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[62], 0.345382, 1e-6)
        XCTAssertEqualWithAccuracy(thetas[63], -0.458212, 1e-6)
    }
    
    func testNet4353()
    {
        let layersizes: [Int32] = [4, 3, 5, 3]
        let arch = netarchitecture(layersizes, Int32(layersizes.count))

        let x: [Float] = [0.645569620253164, 0.795454545454545, 0.202898550724638, 0.04]
        let y: [Float] = [1, 0, 0]
        var thetas: [Float] = [0.31127269008527, -0.43959294610575, -0.491446680012708, -0.664910481694918, 0.41552354924162, -0.449353455229351, -0.693405389243006, 0.0239283603008561, 0.0103757485329294, 0.0421512486018777, -0.604280611717333, -0.332863456899268, -0.613940519320367, 0.483742036575401, -0.220401448606618, -0.415268473472035, 0.133708873260175, 0.355333912584365, 0.365730477716817, 0.0284293866683214, 0.0302261915477985, -0.177586279009537, 0.547643201351005, -0.610095383193791, -0.543208448082842, -0.80485141742137, -0.602419256249932, 0.069231079069165, 0.703581138355337, -0.794128032984584, 0.544101895488743, -0.709945784348422, -0.789560317692621, 0.33357518507195, -0.332207928837549, 0.144037841520174, -0.199924230499394, -0.666534076770724, 0.556101155770297, -0.60774359516734, -0.433080549399777, -0.859175383856612, -0.820404026954991, 0.471859699177925, 0.0618723549699592, -0.777675208292327, -0.364411921718135, 0.528595001959508, 0.487819840988319, 0.256596238540938, 0.430170280223302, 0.219028882675686, -0.617222393828169]
    
        var activations = [Float](count: Int(arch.units), repeatedValue: 0.0)
        
        forwardprop(arch, thetas, x, &activations)
        let netActivation = outputactivations(arch, &activations)

        XCTAssertEqualWithAccuracy(activations[0], 0.645570, 1e-6)
        XCTAssertEqualWithAccuracy(activations[1], 0.795455, 1e-6)
        XCTAssertEqualWithAccuracy(activations[2], 0.202899, 1e-6)
        XCTAssertEqualWithAccuracy(activations[3], 0.040000, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[4], 0.335024, 1e-6)
        XCTAssertEqualWithAccuracy(activations[5], 0.436454, 1e-6)
        XCTAssertEqualWithAccuracy(activations[6], 0.299509, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[7], 0.355842, 1e-6)
        XCTAssertEqualWithAccuracy(activations[8], 0.400988, 1e-6)
        XCTAssertEqualWithAccuracy(activations[9], 0.582378, 1e-6)
        XCTAssertEqualWithAccuracy(activations[10], 0.638376, 1e-6)
        XCTAssertEqualWithAccuracy(activations[11], 0.354422, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[12], 0.627928, 1e-6)
        XCTAssertEqualWithAccuracy(activations[13], 0.308100, 1e-6)
        XCTAssertEqualWithAccuracy(activations[14], 0.289336, 1e-6)
        
        
        //Gradients
        var grad = [Float](count: Int(arch.parameterscount), repeatedValue: 0.0)
        vanillabackprop(arch, thetas, activations, y, &grad)

        XCTAssertEqualWithAccuracy(grad[0], -0.007023, 1e-6)
        XCTAssertEqualWithAccuracy(grad[1], 0.028038, 1e-6)
        XCTAssertEqualWithAccuracy(grad[2], -0.001267, 1e-6)
        XCTAssertEqualWithAccuracy(grad[3], -0.004534, 1e-6)
        XCTAssertEqualWithAccuracy(grad[4], 0.018101, 1e-6)
        XCTAssertEqualWithAccuracy(grad[5], -0.000818, 1e-6)
        XCTAssertEqualWithAccuracy(grad[6], -0.005586, 1e-6)
        XCTAssertEqualWithAccuracy(grad[7], 0.022303, 1e-6)
        XCTAssertEqualWithAccuracy(grad[8], -0.001008, 1e-6)
        XCTAssertEqualWithAccuracy(grad[9], -0.001425, 1e-6)
        XCTAssertEqualWithAccuracy(grad[10], 0.005689, 1e-6)
        XCTAssertEqualWithAccuracy(grad[11], -0.000257, 1e-6)
        XCTAssertEqualWithAccuracy(grad[12], -0.000281, 1e-6)
        XCTAssertEqualWithAccuracy(grad[13], 0.001122, 1e-6)
        XCTAssertEqualWithAccuracy(grad[14], -0.000051, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[15], -0.119070, 1e-6)
        XCTAssertEqualWithAccuracy(grad[16], 0.048864, 1e-6)
        XCTAssertEqualWithAccuracy(grad[17], -0.089517, 1e-6)
        XCTAssertEqualWithAccuracy(grad[18], 0.006433, 1e-6)
        XCTAssertEqualWithAccuracy(grad[19], -0.062042, 1e-6)
        XCTAssertEqualWithAccuracy(grad[20], -0.039891, 1e-6)
        XCTAssertEqualWithAccuracy(grad[21], 0.016371, 1e-6)
        XCTAssertEqualWithAccuracy(grad[22], -0.029990, 1e-6)
        XCTAssertEqualWithAccuracy(grad[23], 0.002155, 1e-6)
        XCTAssertEqualWithAccuracy(grad[24], -0.020786, 1e-6)
        XCTAssertEqualWithAccuracy(grad[25], -0.051969, 1e-6)
        XCTAssertEqualWithAccuracy(grad[26], 0.021327, 1e-6)
        XCTAssertEqualWithAccuracy(grad[27], -0.039070, 1e-6)
        XCTAssertEqualWithAccuracy(grad[28], 0.002808, 1e-6)
        XCTAssertEqualWithAccuracy(grad[29], -0.027079, 1e-6)
        XCTAssertEqualWithAccuracy(grad[30], -0.035663, 1e-6)
        XCTAssertEqualWithAccuracy(grad[31], 0.014635, 1e-6)
        XCTAssertEqualWithAccuracy(grad[32], -0.026811, 1e-6)
        XCTAssertEqualWithAccuracy(grad[33], 0.001927, 1e-6)
        XCTAssertEqualWithAccuracy(grad[34], -0.018582, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[35], -0.372072, 1e-6)
        XCTAssertEqualWithAccuracy(grad[36], 0.308100, 1e-6)
        XCTAssertEqualWithAccuracy(grad[37], 0.289336, 1e-6)
        XCTAssertEqualWithAccuracy(grad[38], -0.132399, 1e-6)
        XCTAssertEqualWithAccuracy(grad[39], 0.109635, 1e-6)
        XCTAssertEqualWithAccuracy(grad[40], 0.102958, 1e-6)
        XCTAssertEqualWithAccuracy(grad[41], -0.149196, 1e-6)
        XCTAssertEqualWithAccuracy(grad[42], 0.123544, 1e-6)
        XCTAssertEqualWithAccuracy(grad[43], 0.116020, 1e-6)
        XCTAssertEqualWithAccuracy(grad[44], -0.216686, 1e-6)
        XCTAssertEqualWithAccuracy(grad[45], 0.179431, 1e-6)
        XCTAssertEqualWithAccuracy(grad[46], 0.168503, 1e-6)
        XCTAssertEqualWithAccuracy(grad[47], -0.237522, 1e-6)
        XCTAssertEqualWithAccuracy(grad[48], 0.196684, 1e-6)
        XCTAssertEqualWithAccuracy(grad[49], 0.184705, 1e-6)
        XCTAssertEqualWithAccuracy(grad[50], -0.131870, 1e-6)
        XCTAssertEqualWithAccuracy(grad[51], 0.109197, 1e-6)
        XCTAssertEqualWithAccuracy(grad[52], 0.102547, 1e-6)
    
        var l2grad = [Float](thetas)
        zerobias(arch, &l2grad)
        
        XCTAssertEqualWithAccuracy(l2grad[0], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[1], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[2], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[3], -0.664910, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[4], 0.415524, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[5], -0.449353, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[6], -0.693405, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[7], 0.023928, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[8], 0.010376, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[9], 0.042151, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[10], -0.604281, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[11], -0.332863, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[12], -0.613941, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[13], 0.483742, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[14], -0.220401, 1e-6)
        
        XCTAssertEqualWithAccuracy(l2grad[15], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[16], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[17], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[18], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[19], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[20], 0.030226, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[21], -0.177586, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[22], 0.547643, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[23], -0.610095, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[24], -0.543208, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[25], -0.804851, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[26], -0.602419, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[27], 0.069231, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[28], 0.703581, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[29], -0.794128, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[30], 0.544102, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[31], -0.709946, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[32], -0.789560, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[33], 0.333575, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[34], -0.332208, 1e-6)
        
        XCTAssertEqualWithAccuracy(l2grad[35], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[36], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[37], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[38], 0.556101, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[39], -0.607744, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[40], -0.433081, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[41], -0.859175, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[42], -0.820404, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[43], 0.471860, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[44], 0.061872, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[45], -0.777675, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[46], -0.364412, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[47], 0.528595, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[48], 0.487820, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[49], 0.256596, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[50], 0.430170, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[51], 0.219029, 1e-6)
        XCTAssertEqualWithAccuracy(l2grad[52], -0.617222, 1e-6)
        
        var l1grad = [Float](l2grad)
        signtheta(&l1grad, Int32(l1grad.count))
        
        XCTAssertEqualWithAccuracy(l1grad[0], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[1], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[2], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[3], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[4], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[5], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[6], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[7], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[8], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[9], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[10], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[11], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[12], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[13], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[14], -1.000000, 1e-6)
        
        XCTAssertEqualWithAccuracy(l1grad[15], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[16], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[17], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[18], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[19], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[20], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[21], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[22], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[23], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[24], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[25], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[26], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[27], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[28], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[29], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[30], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[31], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[32], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[33], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[34], -1.000000, 1e-6)
        
        XCTAssertEqualWithAccuracy(l1grad[35], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[36], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[37], 0.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[38], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[39], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[40], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[41], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[42], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[43], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[44], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[45], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[46], -1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[47], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[48], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[49], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[50], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[51], 1.000000, 1e-6)
        XCTAssertEqualWithAccuracy(l1grad[52], -1.000000, 1e-6)
        
        regularization(arch, &grad, thetas, 0.1, 0.3)

        XCTAssertEqualWithAccuracy(grad[0], -0.007023, 1e-6)
        XCTAssertEqualWithAccuracy(grad[1], 0.028038, 1e-6)
        XCTAssertEqualWithAccuracy(grad[2], -0.001267, 1e-6)
        XCTAssertEqualWithAccuracy(grad[3], -0.304007, 1e-6)
        XCTAssertEqualWithAccuracy(grad[4], 0.242758, 1e-6)
        XCTAssertEqualWithAccuracy(grad[5], -0.235624, 1e-6)
        XCTAssertEqualWithAccuracy(grad[6], -0.313608, 1e-6)
        XCTAssertEqualWithAccuracy(grad[7], 0.129482, 1e-6)
        XCTAssertEqualWithAccuracy(grad[8], 0.102105, 1e-6)
        XCTAssertEqualWithAccuracy(grad[9], 0.111220, 1e-6)
        XCTAssertEqualWithAccuracy(grad[10], -0.275595, 1e-6)
        XCTAssertEqualWithAccuracy(grad[11], -0.200116, 1e-6)
        XCTAssertEqualWithAccuracy(grad[12], -0.284463, 1e-6)
        XCTAssertEqualWithAccuracy(grad[13], 0.246244, 1e-6)
        XCTAssertEqualWithAccuracy(grad[14], -0.166171, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[15], -0.119070, 1e-6)
        XCTAssertEqualWithAccuracy(grad[16], 0.048864, 1e-6)
        XCTAssertEqualWithAccuracy(grad[17], -0.089517, 1e-6)
        XCTAssertEqualWithAccuracy(grad[18], 0.006433, 1e-6)
        XCTAssertEqualWithAccuracy(grad[19], -0.062042, 1e-6)
        XCTAssertEqualWithAccuracy(grad[20], 0.069177, 1e-6)
        XCTAssertEqualWithAccuracy(grad[21], -0.136905, 1e-6)
        XCTAssertEqualWithAccuracy(grad[22], 0.234303, 1e-6)
        XCTAssertEqualWithAccuracy(grad[23], -0.280874, 1e-6)
        XCTAssertEqualWithAccuracy(grad[24], -0.283748, 1e-6)
        XCTAssertEqualWithAccuracy(grad[25], -0.393424, 1e-6)
        XCTAssertEqualWithAccuracy(grad[26], -0.259399, 1e-6)
        XCTAssertEqualWithAccuracy(grad[27], 0.081699, 1e-6)
        XCTAssertEqualWithAccuracy(grad[28], 0.313882, 1e-6)
        XCTAssertEqualWithAccuracy(grad[29], -0.365317, 1e-6)
        XCTAssertEqualWithAccuracy(grad[30], 0.227568, 1e-6)
        XCTAssertEqualWithAccuracy(grad[31], -0.298348, 1e-6)
        XCTAssertEqualWithAccuracy(grad[32], -0.363679, 1e-6)
        XCTAssertEqualWithAccuracy(grad[33], 0.201999, 1e-6)
        XCTAssertEqualWithAccuracy(grad[34], -0.218245, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[35], -0.372072, 1e-6)
        XCTAssertEqualWithAccuracy(grad[36], 0.308100, 1e-6)
        XCTAssertEqualWithAccuracy(grad[37], 0.289336, 1e-6)
        XCTAssertEqualWithAccuracy(grad[38], 0.134432, 1e-6)
        XCTAssertEqualWithAccuracy(grad[39], -0.172688, 1e-6)
        XCTAssertEqualWithAccuracy(grad[40], -0.126966, 1e-6)
        XCTAssertEqualWithAccuracy(grad[41], -0.506949, 1e-6)
        XCTAssertEqualWithAccuracy(grad[42], -0.222577, 1e-6)
        XCTAssertEqualWithAccuracy(grad[43], 0.357578, 1e-6)
        XCTAssertEqualWithAccuracy(grad[44], -0.098125, 1e-6)
        XCTAssertEqualWithAccuracy(grad[45], -0.153872, 1e-6)
        XCTAssertEqualWithAccuracy(grad[46], -0.040821, 1e-6)
        XCTAssertEqualWithAccuracy(grad[47], 0.021057, 1e-6)
        XCTAssertEqualWithAccuracy(grad[48], 0.443030, 1e-6)
        XCTAssertEqualWithAccuracy(grad[49], 0.361684, 1e-6)
        XCTAssertEqualWithAccuracy(grad[50], 0.097181, 1e-6)
        XCTAssertEqualWithAccuracy(grad[51], 0.274906, 1e-6)
        XCTAssertEqualWithAccuracy(grad[52], -0.182620, 1e-6)
    }

    func testSteepestDecentWithAdaptiveLearning()
    {
        let learningRate: Float = 0.5
        let memory: Float = 0.2
        let alpha: Float = 0.2
        let beta: Float = 0.5
        
        //starts at zero
        var gradientRunningAverage: [Float] = [1, -1]
        let currentGradient: [Float] = [1, 2]
        
        let newLearningRate = adaptiverate(currentGradient, memory, alpha, beta, learningRate, 2, &gradientRunningAverage)
        
        XCTAssertEqualWithAccuracy(gradientRunningAverage[0], 1, 1e-6)
        XCTAssertEqualWithAccuracy(gradientRunningAverage[1], -0.400000006, 1e-6)
        XCTAssertEqualWithAccuracy(newLearningRate, 0.503851652, 1e-6)
    }

    func testNet4543()
    {
        let layersizes: [Int32] = [4, 5, 4, 3]
        let arch = netarchitecture(layersizes, Int32(layersizes.count))

        let x: [Float] = [0.645569620253164, 0.795454545454545, 0.202898550724638, 0.04]
        let y: [Float] = [1, 0, 0]
    
        var thetas: [Float] = [0.274516709297235, -0.387684537831344, -0.433415232653988, -0.391518871473416, -0.167429949511106, -0.586395926228306, 0.366457325061409, -0.396292497768153, 0.0284975266838558, -0.567966321615577, -0.611526072562978, 0.0211028302125387, 0.00915055009475814, -0.758821193475008, -0.669343304516762, 0.0371739070838091, -0.532925406900682, -0.293557975832243, 0.512984186608726, 0.33501202556527, -0.541444644635845, 0.426620375828807, -0.194375807203831, 0.126061934649443, 0.516322961794674, 0.0652716873048915, 0.026803482797526, -0.810038320204754, -0.7734843343635, -0.74440460639501, -0.512141836316264, 0.0583338156896525, -0.73319921779222, 0.344814001173543, -0.748711089671645, 0.498364147182513, 0.459920956746912, -0.575203443502579, -0.313208638993292, 0.405568429614419, 0.206502410954253, 0.663342658728012, 0.135800179315184, -0.188490372146165, -0.628414354101992, 0.314497700533244, 0.524297197694471, -0.572986156487358, -0.408312257707438, 0.504439236842861, 0.193606173462437, 0.237868443768336, -0.389572731064328, 0.0294658646219818, -0.521337561593108, 0.274312917530031, -0.831766613405672, -0.901278877250215, -0.659838494042704, -0.147453220554047, -0.24967234028206, -0.843425153116835, 0.382516139902567, -0.4306203872368]
        
        var activations = [Float](count: Int(arch.units), repeatedValue: 0.0)
        
        forwardprop(arch, thetas, x, &activations)
        
        XCTAssertEqualWithAccuracy(activations[0], 0.645570, 1e-6)
        XCTAssertEqualWithAccuracy(activations[1], 0.795455, 1e-6)
        XCTAssertEqualWithAccuracy(activations[2], 0.202899, 1e-6)
        XCTAssertEqualWithAccuracy(activations[3], 0.040000, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[4], 0.353292, 1e-6)
        XCTAssertEqualWithAccuracy(activations[5], 0.443890, 1e-6)
        XCTAssertEqualWithAccuracy(activations[6], 0.320972, 1e-6)
        XCTAssertEqualWithAccuracy(activations[7], 0.295753, 1e-6)
        XCTAssertEqualWithAccuracy(activations[8], 0.273309, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[9], 0.513211, 1e-6)
        XCTAssertEqualWithAccuracy(activations[10], 0.400460, 1e-6)
        XCTAssertEqualWithAccuracy(activations[11], 0.342903, 1e-6)
        XCTAssertEqualWithAccuracy(activations[12], 0.257399, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[13], 0.492751, 1e-6)
        XCTAssertEqualWithAccuracy(activations[14], 0.480893, 1e-6)
        XCTAssertEqualWithAccuracy(activations[15], 0.357308, 1e-6)
        
        var grad = [Float](count: Int(arch.parameterscount), repeatedValue: 0.0)
        vanillabackprop(arch, thetas, activations, y, &grad)
        
        XCTAssertEqualWithAccuracy(grad[0], 0.008972, 1e-6)
        XCTAssertEqualWithAccuracy(grad[1], 0.053532, 1e-6)
        XCTAssertEqualWithAccuracy(grad[2], 0.020732, 1e-6)
        XCTAssertEqualWithAccuracy(grad[3], -0.017969, 1e-6)
        XCTAssertEqualWithAccuracy(grad[4], -0.032707, 1e-6)
        XCTAssertEqualWithAccuracy(grad[5], 0.005792, 1e-6)
        XCTAssertEqualWithAccuracy(grad[6], 0.034559, 1e-6)
        XCTAssertEqualWithAccuracy(grad[7], 0.013384, 1e-6)
        XCTAssertEqualWithAccuracy(grad[8], -0.011600, 1e-6)
        XCTAssertEqualWithAccuracy(grad[9], -0.021115, 1e-6)
        XCTAssertEqualWithAccuracy(grad[10], 0.007137, 1e-6)
        XCTAssertEqualWithAccuracy(grad[11], 0.042583, 1e-6)
        XCTAssertEqualWithAccuracy(grad[12], 0.016491, 1e-6)
        XCTAssertEqualWithAccuracy(grad[13], -0.014293, 1e-6)
        XCTAssertEqualWithAccuracy(grad[14], -0.026017, 1e-6)
        XCTAssertEqualWithAccuracy(grad[15], 0.001820, 1e-6)
        XCTAssertEqualWithAccuracy(grad[16], 0.010862, 1e-6)
        XCTAssertEqualWithAccuracy(grad[17], 0.004206, 1e-6)
        XCTAssertEqualWithAccuracy(grad[18], -0.003646, 1e-6)
        XCTAssertEqualWithAccuracy(grad[19], -0.006636, 1e-6)
        XCTAssertEqualWithAccuracy(grad[20], 0.000359, 1e-6)
        XCTAssertEqualWithAccuracy(grad[21], 0.002141, 1e-6)
        XCTAssertEqualWithAccuracy(grad[22], 0.000829, 1e-6)
        XCTAssertEqualWithAccuracy(grad[23], -0.000719, 1e-6)
        XCTAssertEqualWithAccuracy(grad[24], -0.001308, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[25], 0.006371, 1e-6)
        XCTAssertEqualWithAccuracy(grad[26], -0.206760, 1e-6)
        XCTAssertEqualWithAccuracy(grad[27], 0.039337, 1e-6)
        XCTAssertEqualWithAccuracy(grad[28], 0.087527, 1e-6)
        XCTAssertEqualWithAccuracy(grad[29], 0.002251, 1e-6)
        XCTAssertEqualWithAccuracy(grad[30], -0.073047, 1e-6)
        XCTAssertEqualWithAccuracy(grad[31], 0.013898, 1e-6)
        XCTAssertEqualWithAccuracy(grad[32], 0.030923, 1e-6)
        XCTAssertEqualWithAccuracy(grad[33], 0.002828, 1e-6)
        XCTAssertEqualWithAccuracy(grad[34], -0.091779, 1e-6)
        XCTAssertEqualWithAccuracy(grad[35], 0.017461, 1e-6)
        XCTAssertEqualWithAccuracy(grad[36], 0.038853, 1e-6)
        XCTAssertEqualWithAccuracy(grad[37], 0.002045, 1e-6)
        XCTAssertEqualWithAccuracy(grad[38], -0.066364, 1e-6)
        XCTAssertEqualWithAccuracy(grad[39], 0.012626, 1e-6)
        XCTAssertEqualWithAccuracy(grad[40], 0.028094, 1e-6)
        XCTAssertEqualWithAccuracy(grad[41], 0.001884, 1e-6)
        XCTAssertEqualWithAccuracy(grad[42], -0.061150, 1e-6)
        XCTAssertEqualWithAccuracy(grad[43], 0.011634, 1e-6)
        XCTAssertEqualWithAccuracy(grad[44], 0.025887, 1e-6)
        XCTAssertEqualWithAccuracy(grad[45], 0.001741, 1e-6)
        XCTAssertEqualWithAccuracy(grad[46], -0.056509, 1e-6)
        XCTAssertEqualWithAccuracy(grad[47], 0.010751, 1e-6)
        XCTAssertEqualWithAccuracy(grad[48], 0.023922, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[49], -0.507249, 1e-6)
        XCTAssertEqualWithAccuracy(grad[50], 0.480893, 1e-6)
        XCTAssertEqualWithAccuracy(grad[51], 0.357308, 1e-6)
        XCTAssertEqualWithAccuracy(grad[52], -0.260326, 1e-6)
        XCTAssertEqualWithAccuracy(grad[53], 0.246800, 1e-6)
        XCTAssertEqualWithAccuracy(grad[54], 0.183374, 1e-6)
        XCTAssertEqualWithAccuracy(grad[55], -0.203133, 1e-6)
        XCTAssertEqualWithAccuracy(grad[56], 0.192578, 1e-6)
        XCTAssertEqualWithAccuracy(grad[57], 0.143087, 1e-6)
        XCTAssertEqualWithAccuracy(grad[58], -0.173937, 1e-6)
        XCTAssertEqualWithAccuracy(grad[59], 0.164900, 1e-6)
        XCTAssertEqualWithAccuracy(grad[60], 0.122522, 1e-6)
        XCTAssertEqualWithAccuracy(grad[61], -0.130565, 1e-6)
        XCTAssertEqualWithAccuracy(grad[62], 0.123781, 1e-6)
        XCTAssertEqualWithAccuracy(grad[63], 0.091971, 1e-6)
    }
    
    func testNet4753()
    {
        let layersizes: [Int32] = [4, 7, 5, 3]
        let arch = netarchitecture(layersizes, Int32(layersizes.count))

        let y: [Float] = [1, 0, 0]
        let x: [Float] = [0.645569620253164, 0.795454545454545, 0.202898550724638, 0.04]
        var thetas: [Float] = [0.248309706387817, -0.350673858820613, -0.392038828673964, -0.354142144096555, -0.151446087599042, 0.0590404626064715, 0.28447387366318, -0.530415072516059, 0.331473122428908, -0.358460051544468, 0.0257769827627702, -0.513744867915598, -0.673339301352121, 0.0242446624129471, -0.553146145148468, 0.0190882281353364, 0.00827698398809013, -0.686379595016258, -0.605443799186917, 0.31189599936922, -0.463249721046725, 0.0336250641241444, -0.48204916798283, -0.265533180014119, 0.464011655554582, 0.303029778833714, -0.520291090968667, -0.67723466204154, -0.489755108455749, 0.385892649421244, -0.175819532950246, 0.114027310248683, 0.467031691341257, 0.600016010846615, -0.283307873628716, 0.117606405125431, -0.669857082988088, 0.209509951447591, 0.181675024834958, -0.0507793485951119, 0.454054692336404, -0.634969148642942, -0.503959974232726, -0.398178139770687, -0.690088807687869, -0.70151376333619, 0.39830323227567, -0.644176601197936, -0.688363112877321, 0.421165989141492, 0.0505185662869183, 0.178836333829117, 0.147869157483728, -0.190690399713218, -0.0860131999068492, 0.431596011795423, -0.544222794755115, 0.0225049258412588, -0.328891753383972, 0.0190865642028868, 0.351232563019048, -0.353608787851219, -0.635272244411427, -0.610005696040316, -0.00057021557975377, -0.163237450647362, 0.385271831056361, -0.112619257442323, 0.308545682308495, -0.694445455639117, -0.496220567534858, -0.29754108813216, 0.292151527607638, -0.606778310992259, 0.465047056024131, -0.843687452294582, 0.742999162862542, -0.620281863279644, -0.86070106486703, -0.0672143766678439, -0.109204621465336, -0.75556716117787, -0.215929678587277, 0.50625468244176, 0.266983717092253, 0.201967656289033, 0.0560588610160539, -0.806202043421753, -0.789046457253427, -0.171994390041727, 0.716227600177918, -0.220373623287948, -0.350961065178847]
    
        
        var activations = [Float](count: Int(arch.units), repeatedValue: 0.0)
        forwardprop(arch, thetas, x, &activations)
        
        XCTAssertEqualWithAccuracy(activations[0], 0.645570, 1e-6)
        XCTAssertEqualWithAccuracy(activations[1], 0.795455, 1e-6)
        XCTAssertEqualWithAccuracy(activations[2], 0.202899, 1e-6)
        XCTAssertEqualWithAccuracy(activations[3], 0.040000, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[4], 0.366588, 1e-6)
        XCTAssertEqualWithAccuracy(activations[5], 0.449208, 1e-6)
        XCTAssertEqualWithAccuracy(activations[6], 0.336758, 1e-6)
        XCTAssertEqualWithAccuracy(activations[7], 0.313291, 1e-6)
        XCTAssertEqualWithAccuracy(activations[8], 0.292238, 1e-6)
        XCTAssertEqualWithAccuracy(activations[9], 0.447911, 1e-6)
        XCTAssertEqualWithAccuracy(activations[10], 0.445934, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[11], 0.482359, 1e-6)
        XCTAssertEqualWithAccuracy(activations[12], 0.289577, 1e-6)
        XCTAssertEqualWithAccuracy(activations[13], 0.422243, 1e-6)
        XCTAssertEqualWithAccuracy(activations[14], 0.320513, 1e-6)
        XCTAssertEqualWithAccuracy(activations[15], 0.439944, 1e-6)
        
        XCTAssertEqualWithAccuracy(activations[16], 0.212786, 1e-6)
        XCTAssertEqualWithAccuracy(activations[17], 0.594711, 1e-6)
        XCTAssertEqualWithAccuracy(activations[18], 0.329109, 1e-6)
        
        var grad = [Float](count: Int(arch.parameterscount), repeatedValue: 0.0)
        vanillabackprop(arch, thetas, activations, y, &grad)
        
        XCTAssertEqualWithAccuracy(grad[0], 0.028488, 1e-6)
        XCTAssertEqualWithAccuracy(grad[1], -0.035299, 1e-6)
        XCTAssertEqualWithAccuracy(grad[2], 0.009146, 1e-6)
        XCTAssertEqualWithAccuracy(grad[3], -0.003879, 1e-6)
        XCTAssertEqualWithAccuracy(grad[4], 0.000719, 1e-6)
        XCTAssertEqualWithAccuracy(grad[5], 0.042926, 1e-6)
        XCTAssertEqualWithAccuracy(grad[6], -0.055748, 1e-6)
        XCTAssertEqualWithAccuracy(grad[7], 0.018391, 1e-6)
        XCTAssertEqualWithAccuracy(grad[8], -0.022788, 1e-6)
        XCTAssertEqualWithAccuracy(grad[9], 0.005905, 1e-6)
        XCTAssertEqualWithAccuracy(grad[10], -0.002504, 1e-6)
        XCTAssertEqualWithAccuracy(grad[11], 0.000464, 1e-6)
        XCTAssertEqualWithAccuracy(grad[12], 0.027712, 1e-6)
        XCTAssertEqualWithAccuracy(grad[13], -0.035989, 1e-6)
        XCTAssertEqualWithAccuracy(grad[14], 0.022661, 1e-6)
        XCTAssertEqualWithAccuracy(grad[15], -0.028079, 1e-6)
        XCTAssertEqualWithAccuracy(grad[16], 0.007276, 1e-6)
        XCTAssertEqualWithAccuracy(grad[17], -0.003086, 1e-6)
        XCTAssertEqualWithAccuracy(grad[18], 0.000572, 1e-6)
        XCTAssertEqualWithAccuracy(grad[19], 0.034146, 1e-6)
        XCTAssertEqualWithAccuracy(grad[20], -0.044345, 1e-6)
        XCTAssertEqualWithAccuracy(grad[21], 0.005780, 1e-6)
        XCTAssertEqualWithAccuracy(grad[22], -0.007162, 1e-6)
        XCTAssertEqualWithAccuracy(grad[23], 0.001856, 1e-6)
        XCTAssertEqualWithAccuracy(grad[24], -0.000787, 1e-6)
        XCTAssertEqualWithAccuracy(grad[25], 0.000146, 1e-6)
        XCTAssertEqualWithAccuracy(grad[26], 0.008710, 1e-6)
        XCTAssertEqualWithAccuracy(grad[27], -0.011311, 1e-6)
        XCTAssertEqualWithAccuracy(grad[28], 0.001140, 1e-6)
        XCTAssertEqualWithAccuracy(grad[29], -0.001412, 1e-6)
        XCTAssertEqualWithAccuracy(grad[30], 0.000366, 1e-6)
        XCTAssertEqualWithAccuracy(grad[31], -0.000155, 1e-6)
        XCTAssertEqualWithAccuracy(grad[32], 0.000029, 1e-6)
        XCTAssertEqualWithAccuracy(grad[33], 0.001717, 1e-6)
        XCTAssertEqualWithAccuracy(grad[34], -0.002230, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[35], 0.150223, 1e-6)
        XCTAssertEqualWithAccuracy(grad[36], 0.130220, 1e-6)
        XCTAssertEqualWithAccuracy(grad[37], -0.017470, 1e-6)
        XCTAssertEqualWithAccuracy(grad[38], 0.023694, 1e-6)
        XCTAssertEqualWithAccuracy(grad[39], -0.199674, 1e-6)
        XCTAssertEqualWithAccuracy(grad[40], 0.055070, 1e-6)
        XCTAssertEqualWithAccuracy(grad[41], 0.047737, 1e-6)
        XCTAssertEqualWithAccuracy(grad[42], -0.006404, 1e-6)
        XCTAssertEqualWithAccuracy(grad[43], 0.008686, 1e-6)
        XCTAssertEqualWithAccuracy(grad[44], -0.073198, 1e-6)
        XCTAssertEqualWithAccuracy(grad[45], 0.067482, 1e-6)
        XCTAssertEqualWithAccuracy(grad[46], 0.058496, 1e-6)
        XCTAssertEqualWithAccuracy(grad[47], -0.007848, 1e-6)
        XCTAssertEqualWithAccuracy(grad[48], 0.010643, 1e-6)
        XCTAssertEqualWithAccuracy(grad[49], -0.089695, 1e-6)
        XCTAssertEqualWithAccuracy(grad[50], 0.050589, 1e-6)
        XCTAssertEqualWithAccuracy(grad[51], 0.043853, 1e-6)
        XCTAssertEqualWithAccuracy(grad[52], -0.005883, 1e-6)
        XCTAssertEqualWithAccuracy(grad[53], 0.007979, 1e-6)
        XCTAssertEqualWithAccuracy(grad[54], -0.067242, 1e-6)
        XCTAssertEqualWithAccuracy(grad[55], 0.047064, 1e-6)
        XCTAssertEqualWithAccuracy(grad[56], 0.040797, 1e-6)
        XCTAssertEqualWithAccuracy(grad[57], -0.005473, 1e-6)
        XCTAssertEqualWithAccuracy(grad[58], 0.007423, 1e-6)
        XCTAssertEqualWithAccuracy(grad[59], -0.062556, 1e-6)
        XCTAssertEqualWithAccuracy(grad[60], 0.043901, 1e-6)
        XCTAssertEqualWithAccuracy(grad[61], 0.038055, 1e-6)
        XCTAssertEqualWithAccuracy(grad[62], -0.005105, 1e-6)
        XCTAssertEqualWithAccuracy(grad[63], 0.006924, 1e-6)
        XCTAssertEqualWithAccuracy(grad[64], -0.058352, 1e-6)
        XCTAssertEqualWithAccuracy(grad[65], 0.067287, 1e-6)
        XCTAssertEqualWithAccuracy(grad[66], 0.058327, 1e-6)
        XCTAssertEqualWithAccuracy(grad[67], -0.007825, 1e-6)
        XCTAssertEqualWithAccuracy(grad[68], 0.010613, 1e-6)
        XCTAssertEqualWithAccuracy(grad[69], -0.089436, 1e-6)
        XCTAssertEqualWithAccuracy(grad[70], 0.066990, 1e-6)
        XCTAssertEqualWithAccuracy(grad[71], 0.058070, 1e-6)
        XCTAssertEqualWithAccuracy(grad[72], -0.007790, 1e-6)
        XCTAssertEqualWithAccuracy(grad[73], 0.010566, 1e-6)
        XCTAssertEqualWithAccuracy(grad[74], -0.089042, 1e-6)
        
        XCTAssertEqualWithAccuracy(grad[75], -0.787214, 1e-6)
        XCTAssertEqualWithAccuracy(grad[76], 0.594711, 1e-6)
        XCTAssertEqualWithAccuracy(grad[77], 0.329109, 1e-6)
        XCTAssertEqualWithAccuracy(grad[78], -0.379720, 1e-6)
        XCTAssertEqualWithAccuracy(grad[79], 0.286864, 1e-6)
        XCTAssertEqualWithAccuracy(grad[80], 0.158749, 1e-6)
        XCTAssertEqualWithAccuracy(grad[81], -0.227959, 1e-6)
        XCTAssertEqualWithAccuracy(grad[82], 0.172215, 1e-6)
        XCTAssertEqualWithAccuracy(grad[83], 0.095303, 1e-6)
        XCTAssertEqualWithAccuracy(grad[84], -0.332396, 1e-6)
        XCTAssertEqualWithAccuracy(grad[85], 0.251113, 1e-6)
        XCTAssertEqualWithAccuracy(grad[86], 0.138964, 1e-6)
        XCTAssertEqualWithAccuracy(grad[87], -0.252313, 1e-6)
        XCTAssertEqualWithAccuracy(grad[88], 0.190613, 1e-6)
        XCTAssertEqualWithAccuracy(grad[89], 0.105484, 1e-6)
        XCTAssertEqualWithAccuracy(grad[90], -0.346330, 1e-6)
        XCTAssertEqualWithAccuracy(grad[91], 0.261640, 1e-6)
        XCTAssertEqualWithAccuracy(grad[92], 0.144790, 1e-6)
    }
}
