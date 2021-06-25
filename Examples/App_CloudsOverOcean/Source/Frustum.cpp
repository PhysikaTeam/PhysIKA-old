#include "Common.h"

//***********************************************************************//
//																		 //
//		- "Talk to me like I'm a 3 year old!" Programming Lessons -		 //
//                                                                       //
//		$Author:		DigiBen		digiben@gametutorials.com			 //
//																		 //
//		$Program:		Frustum Culling									 //
//																		 //
//		$Description:	Demonstrates checking if shapes are in view		 //
//																		 //
//		$Date:			8/28/01											 //
//																		 //
//***********************************************************************//

#define FIRST_OBJECT_ID  3								

enum FrustumSide
{
	RIGHT	= 0,		// The RIGHT side of the frustum
	LEFT	= 1,		// The LEFT	 side of the frustum
	BOTTOM	= 2,		// The BOTTOM side of the frustum
	TOP		= 3,		// The TOP side of the frustum
	BACK	= 4,		// The BACK	side of the frustum
	FRONT	= 5			// The FRONT side of the frustum
}; 

enum PlaneData
{
	A = 0,				// The X value of the plane's normal
	B = 1,				// The Y value of the plane's normal
	C = 2,				// The Z value of the plane's normal
	D = 3				// The distance the plane is from the origin
};

void NormalizePlane(float frustum[6][4], int side)
{
	float magnitude = (float)sqrt( frustum[side][A] * frustum[side][A] + 
								   frustum[side][B] * frustum[side][B] + 
								   frustum[side][C] * frustum[side][C] );

	frustum[side][A] /= magnitude;
	frustum[side][B] /= magnitude;
	frustum[side][C] /= magnitude;
	frustum[side][D] /= magnitude; 
}


void CFrustum::CalculateFrustum()
{    
	float   proj[16];								// This will hold our projection matrix
	float   modl[16];								// This will hold our modelview matrix
	float   clip[16];								// This will hold the clipping planes

	glGetFloatv( GL_PROJECTION_MATRIX, proj );
	glGetFloatv( GL_MODELVIEW_MATRIX, modl );

	clip[ 0] = modl[ 0] * proj[ 0] + modl[ 1] * proj[ 4] + modl[ 2] * proj[ 8] + modl[ 3] * proj[12];
	clip[ 1] = modl[ 0] * proj[ 1] + modl[ 1] * proj[ 5] + modl[ 2] * proj[ 9] + modl[ 3] * proj[13];
	clip[ 2] = modl[ 0] * proj[ 2] + modl[ 1] * proj[ 6] + modl[ 2] * proj[10] + modl[ 3] * proj[14];
	clip[ 3] = modl[ 0] * proj[ 3] + modl[ 1] * proj[ 7] + modl[ 2] * proj[11] + modl[ 3] * proj[15];

	clip[ 4] = modl[ 4] * proj[ 0] + modl[ 5] * proj[ 4] + modl[ 6] * proj[ 8] + modl[ 7] * proj[12];
	clip[ 5] = modl[ 4] * proj[ 1] + modl[ 5] * proj[ 5] + modl[ 6] * proj[ 9] + modl[ 7] * proj[13];
	clip[ 6] = modl[ 4] * proj[ 2] + modl[ 5] * proj[ 6] + modl[ 6] * proj[10] + modl[ 7] * proj[14];
	clip[ 7] = modl[ 4] * proj[ 3] + modl[ 5] * proj[ 7] + modl[ 6] * proj[11] + modl[ 7] * proj[15];

	clip[ 8] = modl[ 8] * proj[ 0] + modl[ 9] * proj[ 4] + modl[10] * proj[ 8] + modl[11] * proj[12];
	clip[ 9] = modl[ 8] * proj[ 1] + modl[ 9] * proj[ 5] + modl[10] * proj[ 9] + modl[11] * proj[13];
	clip[10] = modl[ 8] * proj[ 2] + modl[ 9] * proj[ 6] + modl[10] * proj[10] + modl[11] * proj[14];
	clip[11] = modl[ 8] * proj[ 3] + modl[ 9] * proj[ 7] + modl[10] * proj[11] + modl[11] * proj[15];

	clip[12] = modl[12] * proj[ 0] + modl[13] * proj[ 4] + modl[14] * proj[ 8] + modl[15] * proj[12];
	clip[13] = modl[12] * proj[ 1] + modl[13] * proj[ 5] + modl[14] * proj[ 9] + modl[15] * proj[13];
	clip[14] = modl[12] * proj[ 2] + modl[13] * proj[ 6] + modl[14] * proj[10] + modl[15] * proj[14];
	clip[15] = modl[12] * proj[ 3] + modl[13] * proj[ 7] + modl[14] * proj[11] + modl[15] * proj[15];
	
	m_Frustum[RIGHT][A] = clip[ 3] - clip[ 0];
	m_Frustum[RIGHT][B] = clip[ 7] - clip[ 4];
	m_Frustum[RIGHT][C] = clip[11] - clip[ 8];
	m_Frustum[RIGHT][D] = clip[15] - clip[12];

	NormalizePlane(m_Frustum, RIGHT);

	m_Frustum[LEFT][A] = clip[ 3] + clip[ 0];
	m_Frustum[LEFT][B] = clip[ 7] + clip[ 4];
	m_Frustum[LEFT][C] = clip[11] + clip[ 8];
	m_Frustum[LEFT][D] = clip[15] + clip[12];

	NormalizePlane(m_Frustum, LEFT);

	m_Frustum[BOTTOM][A] = clip[ 3] + clip[ 1];
	m_Frustum[BOTTOM][B] = clip[ 7] + clip[ 5];
	m_Frustum[BOTTOM][C] = clip[11] + clip[ 9];
	m_Frustum[BOTTOM][D] = clip[15] + clip[13];

	NormalizePlane(m_Frustum, BOTTOM);

	m_Frustum[TOP][A] = clip[ 3] - clip[ 1];
	m_Frustum[TOP][B] = clip[ 7] - clip[ 5];
	m_Frustum[TOP][C] = clip[11] - clip[ 9];
	m_Frustum[TOP][D] = clip[15] - clip[13];

	NormalizePlane(m_Frustum, TOP);

	m_Frustum[BACK][A] = clip[ 3] - clip[ 2];
	m_Frustum[BACK][B] = clip[ 7] - clip[ 6];
	m_Frustum[BACK][C] = clip[11] - clip[10];
	m_Frustum[BACK][D] = clip[15] - clip[14];

	NormalizePlane(m_Frustum, BACK);

	m_Frustum[FRONT][A] = clip[ 3] + clip[ 2];
	m_Frustum[FRONT][B] = clip[ 7] + clip[ 6];
	m_Frustum[FRONT][C] = clip[11] + clip[10];
	m_Frustum[FRONT][D] = clip[15] + clip[14];

	NormalizePlane(m_Frustum, FRONT);
}


bool CFrustum::PointInFrustum( float x, float y, float z, float tolerance )
{	
	if(m_Frustum[0][A] * x + m_Frustum[0][B] * y + m_Frustum[0][C] * z + m_Frustum[0][D] < -tolerance) return false;
	if(m_Frustum[1][A] * x + m_Frustum[1][B] * y + m_Frustum[1][C] * z + m_Frustum[1][D] < -tolerance) return false;
	if(m_Frustum[2][A] * x + m_Frustum[2][B] * y + m_Frustum[2][C] * z + m_Frustum[2][D] < -tolerance) return false;
	if(m_Frustum[3][A] * x + m_Frustum[3][B] * y + m_Frustum[3][C] * z + m_Frustum[3][D] < -tolerance) return false;
	if(m_Frustum[4][A] * x + m_Frustum[4][B] * y + m_Frustum[4][C] * z + m_Frustum[4][D] < -tolerance) return false;
	if(m_Frustum[5][A] * x + m_Frustum[5][B] * y + m_Frustum[5][C] * z + m_Frustum[5][D] < -tolerance) return false;

	return true;
}

bool CFrustum::SphereInFrustum(Vector3 v, float r)
{
	r *= 1.5f;
	if(m_Frustum[0][A] * v.x + m_Frustum[0][B] * v.y + m_Frustum[0][C] * v.z + m_Frustum[0][D] <= -r) return false;
	if(m_Frustum[1][A] * v.x + m_Frustum[1][B] * v.y + m_Frustum[1][C] * v.z + m_Frustum[1][D] <= -r) return false;
	if(m_Frustum[2][A] * v.x + m_Frustum[2][B] * v.y + m_Frustum[2][C] * v.z + m_Frustum[2][D] <= -r) return false;
	if(m_Frustum[3][A] * v.x + m_Frustum[3][B] * v.y + m_Frustum[3][C] * v.z + m_Frustum[3][D] <= -r) return false;
	if(m_Frustum[4][A] * v.x + m_Frustum[4][B] * v.y + m_Frustum[4][C] * v.z + m_Frustum[4][D] <= -r) return false;
	if(m_Frustum[5][A] * v.x + m_Frustum[5][B] * v.y + m_Frustum[5][C] * v.z + m_Frustum[5][D] <= -r) return false;

	return true;	
}
