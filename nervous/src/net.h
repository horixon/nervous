#pragma once
#include "netarch.h"
#include "nettools.h"

template<typename T, typename DIFFERENTIATOR>
void backprop(const NetArch &netarch, const T *thetas, const T *activations, const T *y, T *derivative,
	DIFFERENTIATOR &differentiator)
{
    auto n = netarch.unitcounts[netarch.indexlayerlasthidden];
    auto m = netarch.unitcounts[netarch.indexlayeroutput];
    
    auto outputthetaindex = netarch.parameterscount - thetacount(n, m);
    
    auto a2 = outputactivations(netarch, activations);
    auto a0 = a2 - n;
    auto a1 = a0;
    
    auto biasgrad = derivative + outputthetaindex;
    auto weightgrad = biasgrad;
    
    auto weights = thetas + outputthetaindex + m;
    
    differentiator.outputgrad(a1, a2, y, n, m, biasgrad, weightgrad + m);
    
    for(auto i = netarch.indexlayeroutput; i > 1; i--)
    {
        n = netarch.unitcounts[i - 1];
        m = netarch.unitcounts[i];
        
        auto p = netarch.unitcounts[i - 2];
        
        T *layerbiasgrad = biasgrad;
        biasgrad -= thetacount(p, n);
        weightgrad = biasgrad + n;
        
        a0 -=p;
        
        differentiator.hiddengrad(a1, a0, layerbiasgrad, weights, n, m, p, biasgrad, weightgrad);
        
        a1 = a0;
        weights -= n * p + m;
    }
}
