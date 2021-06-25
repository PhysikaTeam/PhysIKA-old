//system
#include <stdio.h>
#include <tchar.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <vector>
using namespace std;

//GL
#include "glew.h"
#include "glut.h"

//The order of includes is very important!
#include "Vector.h"
#include "Color.h"
// #include "Matrix.h"
#include "Camera.h"
#include "GLText.h"
#include "VolumeClouds.h"
#include "Frustum.h"
#include "SkyDome.h"
#include <IL/il.h>


#define VIEWWIDTH   1200
#define VIEWHEIGHT  900

#define FOV 45
#define FOCAL_LENGTH 1

//const
#define MAXVAL        999999999
#define F_ZERO        (1.0 / 999999999)
#define M_PI          3.14159265358979323846
#define PI            3.141592653589f
#define SUN_AZIMUTH   ( 94.46/180*M_PI)
#define SUN_ELEVATION ( 66.89/180*M_PI)


#define SQR(x) ( (x) * (x) )
#define SUN_COLOR     Color4(1.0,1.0,1.0,1.0)

#define AMBIENT_R     0.3
#define AMBIENT_G     0.3
#define AMBIENT_B     0.3
#define SUN_INTENSITY 1.0


#ifndef PSD 
#define PSD
	inline string get_project_source_dir()
	{
		#if defined PROJECT_SOURCE_DIR
			return string(PROJECT_SOURCE_DIR);
		#else
			return "";
		#endif
	}
#endif


#ifndef  _CARMACK 
#define  _CARMACK

inline float __fastcall carmack_func(float x)
{
	int carmack;
	float isx, halfx;	//Inverse Squareroot of x

	halfx = 0.5f*x;
	carmack = *(int*)&x; 
	carmack = 0x5f3759df - (carmack>>1); 
	isx = *(float*)&carmack; 

	isx = isx*(1.5f-halfx*isx*isx);  //Newton-Rhapson step, add more for accuracy

	return isx;
}
#endif

#ifndef _SIMULATIONSPACE 
#define _SIMULATIONSPACE 

struct SimulationSpace
{
	float  x_min;
	float  x_max;
	float  y_min;
	float  y_max;
	float  z_min;
	float  z_max;

	SimulationSpace(float x_min,float x_max,float y_min, float y_max,float z_min,float z_max)
	{
		this->x_min=x_min;
		this->x_max=x_max;
		this->y_min=y_min;
		this->y_max=y_max;
		this->z_min=z_min;
		this->z_max=z_max;
	}

	SimulationSpace(void)
	{
		this->x_min=0;
		this->x_max=64;
		this->y_min=0;
		this->y_max=64;
		this->z_min=0;
		this->z_max=64;
	}

	Vector3 GetCenter()
	{
		Vector3 Center=Vector3((x_max+x_min)/2,(y_max+y_min)/2,(z_max+z_min)/2);
		return Center;
	}
};
#endif




