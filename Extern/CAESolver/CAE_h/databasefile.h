#pragma once
#include<string>

typedef struct Database
{
	enum Type
	{
		ABSTAT, DEFORC, GLSTAT, JNTFORC, MATSUM, NODOUT, RCFORC, RWFORC, SLEOUT, BINARY, EXTENT, HISTORY
	};
	Type type;

	/**
	ABSTAT,DEFORC,GLSTAT, JNTFORC, MATSUM, NODOUT, RCFORC, RWFORC, SLEOUT
	*/
	double DT;
	int binary;
	int LCUR;
	int IOOPT;
	/**
	NODOUT
	*/
	double DTHF;
	int BINHF;
	/**
	BINARY_D3PLOT
	*/
	double DTorCYCL;
	int LCDT;
	int BEAM;
	int NPLTC;
	int PSETID;
	int CID;
	//Additional Card for D3PLOT option
	//int IOOPT;
	double RATE;
	double CUTOFF;
	double WINDOW;
	int TYPE;
	int PSET;
	/**
	BINARY_D3THDT,BINARY_INTFOR
	*/
	//double DTorCYCL;
	int LCID;
	/**
	BINARY_RUNRSF
	*/
	int LCDTorNR;
	/**
	EXTENT_BINARY
	*/
	int NEIPH;
	int NEIPS;
	int MAXINT0;
	int  STRFLG;
	int SIGFLG;
	int EPSFLG;
	int RLTFLG;
	int ENGFLG;
	int CMPFLG;
	int IEVERP;
	int BEAMIP;
	int DCOMP;
	int SHGE;
	int STSSZ;
	int N3THDT;
	int IALEMAT;
	int NINTSLD;
	int PKP_SEN;
	double SCLP;
	int HYDRO;
	int MSSCL;
	int THERM;
	string INTOUT;
	string nodout;

} Database;