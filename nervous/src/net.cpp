#include <cstring>

#include "nervous.h"
#include "nettools.h"
#include "training.h"
#include "net.h"

extern "C" void forwardprop(NetArch netarch, const float* thetas, const float* x,float* activations) 
{
    memcpy(activations, x, memorysizeactivations(netarch));

    auto *a0 = activations;
    auto *a1 = a0 + netarch.inputunits;

    enumaratethetas(thetas, &netarch, [&a0, &a1](const float *theta, int n, int m, int units) {      
        activation(a0, theta, theta + m, n, m, a1);    
        a0 = a1;
        a1 += m;
    });
}

extern "C" void vanillabackprop(NetArch netarch, const float *thetas, const float *activations, const float *y, float *derivative)
{
	vanillagrad v;
	backprop(netarch, thetas, activations, y, derivative, v);
}
