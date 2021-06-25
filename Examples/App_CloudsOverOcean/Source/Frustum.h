#ifndef _FRUSTUM_H
#define _FRUSTUM_H

class CFrustum 
{
public:

	void CalculateFrustum();

	bool PointInFrustum(float x, float y, float z, float tolerance);
	bool SphereInFrustum(Vector3, float r);

private:

	float m_Frustum[6][4];
};


#endif


