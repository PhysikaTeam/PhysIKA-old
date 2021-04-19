//system
#include <stdio.h>
#include <tchar.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include "Vector.h"
#include <Vector>
using namespace std;
//GL
//#include "GL/glut.h"

#include<CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Polygon_mesh_processing/remesh.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>

//opencv
#include<opencv/cxcore.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
//const
#define MAXVAL 999999999
#define F_ZERO (1.0 / 999999999)
#define M_PI       3.14159265358979323846
//the pixel number to estimate the sun direction
#define RANDCOUNT  10000  

#define  CloudINTERIOR_RES  350
#define   INT_RES   50
#define  CONSTANT_ATTEN  0.35
#define  SUN_INTENSITY (35) 
#define  AMBIENT 0
#define  SOLID_ANGLE   ((10*M_PI)/180.0)


//1
//#define  SUN_INTENSITY (40) 
#define  CLOUD_FILE   "input\\test.jpg"
#define  PIXELTYPEFILE "input/pixelifo_test.txt"

//2
//#define  SUN_INTENSITY (30) 
//#define  CLOUD_FILE   "input\\test3.jpg"
//#define  PIXELTYPEFILE "input\\pixelifo_test3.txt"

//3
//#define  SUN_INTENSITY (40) 
//#define  CLOUD_FILE   "cloudimg\\test0.jpg"

//4
//#define  SUN_INTENSITY (30) 
//#define  CLOUD_FILE   "cloudimg\\cloud14.jpg"

////5
//#define  SUN_INTENSITY (25) 
//#define  CLOUD_FILE   "cloudimg\\shuyun.jpg"



//#define  CLOUD_FILE   "input\\64.jpg"
//#define  CLOUD_FILE   "input\\65.jpg"
//#define  CLOUD_FILE   "input\\61.jpg"
//#define  CLOUD_FILE   "input\\69.jpg"


//#define  CLOUD_FILE   "cloudimg\\test5.jpg"
//#define  CLOUD_FILE   "input\\test1.jpg"
//#define  PIXELTYPEFILE "input\\pixelifo1.txt"
//#define  CLOUD_FILE   "cloudimg\\evaluation.jpg"  


#ifndef   __CYLINDER__
#define  __CYLINDER__

//The cloud volume is the union of cylinders
struct Cylinder//Ô²ÖùÌåÄÚµã
{
	Vector3 center;
	float   radius;
	float   height;

	Cylinder(Vector3 center, float radius, float height)
	{
		this->center = center;
		this->radius = radius;
		this->height = height;
	}

	Cylinder(const  Cylinder&  other)
	{
		this->center = other.center;
		this->radius = other.radius;
		this->height = other.height;
	}

	Cylinder&  operator=(const Cylinder&  other)
	{
		this->center = other.center;
		this->radius = other.radius;
		this->height = other.height;

		return *this;
	}
	Cylinder()
	{
		center = Vector3(0, 0, 0);
		radius = 0;
		height = 0;

	}

};

typedef  vector<Cylinder>   CloudVolume;

#endif