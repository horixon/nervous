//
//  netarch.cpp
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
#include "netarch.h"
#include "nettools.h"

extern "C" NetArch netarchitecture(const int* unitcounts,int depth) {
	int indexoutput = depth - 1;
    int indexlasthidden = depth - 2;
    int inputunits = unitcounts[0];
    int outputunits = unitcounts[indexoutput];

    int parameterscount = 0;
    int units = inputunits;
    
    for(int i=0;i < indexoutput;i++)
    {
        int n = unitcounts[i];
        int m = unitcounts[i+1];
        
        parameterscount += thetacount(n,m);
        units += m;
    }

	return {unitcounts,depth,units,inputunits,outputunits,parameterscount,indexoutput,indexlasthidden};
}

extern "C" int memorysizethetas(NetArch net) {
	return sizeof(float) * net.parameterscount;
}

extern "C" int memorysizeactivations(NetArch net) {
	return sizeof(float) * net.units;
}

extern "C" int memorysizegradient(NetArch net) {
	return memorysizethetas(net);
}

extern "C" int outputactivationsindex(NetArch netarch)
{
    return netarch.units - netarch.outputunits;
}

extern "C" const float* outputactivations(NetArch netarch, const float *activations)
{
    return activations + outputactivationsindex(netarch);
}