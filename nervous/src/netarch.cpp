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