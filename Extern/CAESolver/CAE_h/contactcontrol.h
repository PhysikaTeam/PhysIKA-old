#pragma once

//ls:2020-03-18
typedef struct ContactControl
{
	double SLSFAC;
	int RWPNAL;
	double DTMIN;
	int ISLCHK;
	int SHLTHK;
	int PENOPT;
	int THKCHG;
	int ORIEN;
	int ENMASS;
	int USRSTR;
	int USRFRC;
	int NSBCS;
	int INTERM;
	double XPENE;
	int SSTHK;
	int ECDT;
	int TIEDPRJ;
	double SFRIC;
	double DFRIC;
	double EDC;
	double VFC;
	double TH;
	double TH_SF;
	double PEN_SF;
	int IGNORE0;
	int FRCENG;
	int SKIPRWG;
	int OUTSEG;
	int SPOTSTP;
	int SPOTDEL;
	double SPOTHIN;
	int ISYM;
	int NSEROD;
	int RWGAPS;
	double RWGDTH;
	double RWKSF;
	int ICOV;
	double SWRADF;
	int ITHOFF;
} ContactControl;