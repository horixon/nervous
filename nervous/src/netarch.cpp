#include "netarch.h"
#include "nettools.h"

extern "C" NetDetail details(NetArch arch) {
	int output = arch.depth - 1;
    int lasthidden = arch.depth - 2;
    int inputunits = arch.unitcounts[0];
    int outputunits = arch.unitcounts[output];

    int parameterscount = 0;
    int units = inputunits;
    
    for(int i=0;i < output;i++)
    {
        int n = arch.unitcounts[i];
        int m = arch.unitcounts[i+1];
        
        parameterscount += thetacount(n,m);
        units += m;
    }

	return {units,inputunits,outputunits,parameterscount,output,lasthidden};
}