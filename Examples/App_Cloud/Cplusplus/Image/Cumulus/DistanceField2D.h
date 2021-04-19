#pragma once
/* Vertex structure */ 
#include "total.h"
#include "Vector.h"
//#include <gl\glut.h>

enum DrawType{InputMesh,ConstraintMesh,CloudPIxelOnBasePlane,Distance,Height, DeformedMesh};

typedef struct 
{ 
	float x, y; 
} vertex_t; 

/* bounding rectangle type */ 
typedef struct 
{ 
	float min_x, min_y, max_x, max_y; 
} rect_t; 


class DistanceField2D
{
public:
	DistanceField2D(void);
public:

	//sample space 
	float x_min;
	float x_max;
	float y_min;
	float y_max;

	//grid resolution
	int x_res;
	int y_res;

	//grid interval
	float dx;
	float dy;

	//polygon 
	vertex_t*  verList;
	int nVer;
	void CreateVerList(float* ptList, int verNumber);
	float GetDistance(int idx, int idy);
	Vector2 GetPos(int idx, int idy);

	//distance field 
	float* disList;
	void CreateDisList();
	void DrawDistance();
	float  InterPolate(float x, float y);

public:
	~DistanceField2D(void);

private:
	void vertices_get_extent(const vertex_t* vl, int np, /* in vertices */ rect_t* rc /* out extent*/);
	int is_same(const vertex_t* l_start, const vertex_t* l_end, /* line l */ const vertex_t* p, const vertex_t* q);
	int is_intersect(const vertex_t* s1_start, const vertex_t* s1_end, const vertex_t* s2_start, const vertex_t* s2_end);
	int pt_in_poly(const vertex_t* vl, int np, /* polygon vl with np vertices */ const vertex_t* v);
	float GetPointDistance(vertex_t p1, vertex_t p2);
	float GetNearestDistance(vertex_t PA, vertex_t PB, vertex_t P3);
};
