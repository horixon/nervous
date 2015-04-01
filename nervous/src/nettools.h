#pragma once
#include "netarch.h"

int thetacount(int n, int m);

template<typename T, typename CALLBACK>
void enumaratethetas(T *thetas, const NetArch *netarch, CALLBACK block)
{
     for(int i = 0; i < netarch->depth - 1; i++)
    {
        int n = netarch->unitcounts[i];
        int m = netarch->unitcounts[i+1];
        int thetaunits = thetacount(n, m);
        
        block(thetas, n, m, thetaunits);
        
        thetas += thetaunits;
    }
}
