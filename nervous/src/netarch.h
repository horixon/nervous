typedef struct NetArch {
	const int* unitcounts;
	int depth;
} NetArch;

typedef struct NetDetail {
	int units;
	int inputunits;
	int outputunits;
	int parameterscount;
	int indexoutputlayer;
	int indexlasthiddenLayer;
} NetDetail;