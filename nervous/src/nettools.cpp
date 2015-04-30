#include "netarch.h"
#include "nettools.h"
#include "blasfuncs.h"

#include <cmath>
#include <cstring>
#include <random>

using namespace std;

uniform_real_distribution<float> distribution(-1.0, 1.0);
minstd_rand0 generatortimer;

extern "C" void seed(long seed)
{
    generatortimer.seed(seed);
}

extern "C" int thetacount(int n, int m) 
{
    return n * m + m;
}

float randomweight(int inputUnitCount)
{
    float w = 1/sqrt(inputUnitCount);
    return w * distribution(generatortimer);
}

void randomweight(float *weights, int columns, int rows)
{
    int matrixcount = rows * columns;
    for(int i = 0; i < matrixcount; i++)
    {
        weights[i] = randomweight(columns);
    }
}

void randomtheta(float* theta, int columns, int rows)
{
	Blas.set(rows, 0.0, theta, 1);

    randomweight(theta + rows, columns, rows);
}

extern "C" void randomizethetas(NetArch netarch, float *thetas)
{
	enumaratethetas(thetas, &netarch, [](float *theta, int n, int m, int units) {      
        randomtheta(theta, n, m);
    });
}

extern "C" void zerobias(NetArch netarch, float *thetas)
{
    enumaratethetas(thetas, &netarch, [](float *theta, int n, int m, int units) {      
        Blas.set(m, 0.0, theta, 1);
    });
}
