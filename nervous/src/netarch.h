#pragma once

typedef struct NetArch {
	const int* unitcounts;
	int depth;
	int units;
	int inputunits;
	int outputunits;
	int parameterscount;
	int indexlayeroutput;
	int indexlayerlasthidden;

} NetArch;