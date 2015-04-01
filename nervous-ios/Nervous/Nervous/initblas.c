//
//  initblas.c
//  Nervous
//
//  Created by justin on 3/31/15.
//  Copyright (c) 2015 Microsoft. All rights reserved.
//

#include "initblas.h"
#include "blasfuncs.h"
#include <Accelerate/Accelerate.h>

void initblas() {
    Blas.set = catlas_sset;
    Blas.gemv = cblas_sgemv;
    Blas.ger = cblas_sger;
    Blas.axpy = cblas_saxpy;
    Blas.nrm2 = cblas_snrm2;
    Blas.vmul = vDSP_vmul;
}