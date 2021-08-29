#pragma once
#include<string>

//ls:2020-03-18
typedef struct Airbag
{
	int ABID;
	string title;
	int SID;
	int SIDTYP;
	int RBID;
	double VSCA;
	double PSCA;
	double VINI;
	double MWD;
	double SPSF;
	double CN;
	double BETA;
	int LCID;
	int LCIDDR;
	int storeNum;
	//ls:2020-04-06
	//AIRBAG_SIMPLE_AIRBAG_MODEL
	double CV;
	double CP;
	double T;
	double MU;
	double AREA;
	double PE;
	double RO;
	int LOU;
	double T_EXT;
	double A;
	double B;
	double MW;
	double GASC;
	//
}Airbag;
//