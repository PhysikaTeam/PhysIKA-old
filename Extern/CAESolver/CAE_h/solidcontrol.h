#pragma once

//ls:2020-03-18
typedef struct SolidControl
{
	int ElementSort;
	//ls:2020-04-06
	int ESORT;
	int FMATRIX;
	int NIPTETS;
	int SWLOCL;
	int PSFAIL;
	int PM[10];
	//
} SolidControl;