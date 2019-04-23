#pragma once
#include "functions.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
using namespace std;
#include <string>
typedef unsigned char byte;
typedef unsigned short word;
class BI2Reader
{
public:
	FILE* BI2FilePnt;
	BI2Reader(string dir);
	BI2Reader(char* dir);
	void GetInfo(bool fileout);
	void PrintInfo();
	//-Structures to be used with the format BINX-020:
	typedef struct{ 
		byte fmt;                    ///<File format.
		byte bitorder;               ///<1:BigEndian 0:LittleEndian.
		bool full;                   ///<1:All data of header and particles, 0:Without... [id,pos,vel] of fixed particles
		byte data2d;                 ///<1:Data for a 2D case, 0:3D Case.
		float dp,h,b,rhop0,gamma;
		float massbound,massfluid;
		unsigned np,nfixed,nmoving,nfloat,nfluidout,nbound,nfluid;
		float time;
		unsigned* Id;
		tfloat3* Pos;
		tfloat3* Vel;
		float* Rhop;
		int* OrderOut;
		unsigned OutCountSaved;
		unsigned NFluidOut;
		unsigned OutCount;
	}StInfoFileBi2;
	StInfoFileBi2 info;
	///Structure that describes the header of binary format files.
	typedef struct{
		char titu[16];               ///<Title of the file "#File BINX-000".
		byte fmt;                    ///<File format.
		byte bitorder;               ///<1:BigEndian 0:LittleEndian.
		byte full;                   ///<1:All data of header and particles, 0:Without... [id,pos,vel] of fixed particles
		byte data2d;                 ///<1:Data for a 2D case, 0:3D Case.
	}StHeadFmtBin;//-sizeof(20)

	typedef struct{  //-They must be all of 4 bytes due to conversion ByteOrder ...
		float h,dp,b,rhop0,gamma;
		float massbound,massfluid;
		int np,nfixed,nmoving,nfloat,nfluidout;
		float timestep;
	}StHeadDatFullBi2;//-sizeof(52)  

	typedef struct{ //-They must be all of 4 bytes due to conversion ByteOrder ...
		int np,nfixed,nmoving,nfloat,nfluidout;
		float timestep;
	}StHeadDatBi2;
	unsigned nbound;
	unsigned nfluid;
};
