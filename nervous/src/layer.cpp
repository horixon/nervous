#include "blasfuncs.h"
#include "mathutil.h"

#include <iostream>
using namespace std;

extern "C" void activation(const float* x, const float *bias, 
	const float *weights, int columns, int rows, float *a1)
{
	cout << "About to call sgemv." << endl << columns << ' ' << rows << endl;
    Blas.gemv(CblasColMajor, CblasNoTrans, rows, columns, 1.0, weights, rows,
     x, 1, 0.0, a1, 1);
    
    for(int j = 0; j < rows; j++)
    {
        a1[j] = sigmoid(a1[j] + bias[j]);
    }
}

extern "C" void columnmajortheta(const float* bias, const float* weights, 
	int columns, int rows, float* result)
{
    memcpy(result, bias, sizeof(float) * rows);
    memcpy(result + rows , weights, sizeof(float) * (columns*rows));
}