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
    Blas.sset = catlas_sset;
    Blas.sgemv = cblas_sgemv;
}