#ifndef _CAMERA_H_
#define _CAMERA_H_
#include "common.h"

class CCamera
{
public:
	CCamera(void);
	void Init();
	void PositionCamera(Vector3 vPosition	,Vector3 vView,Vector3 vUpVector);
public:
	~CCamera(void);

public:
	//key control
	bool  key_up;
	bool  key_down;
	bool  key_left;
	bool  key_right;
	bool  key_w;
	bool  key_s;
	bool  key_a;
	bool  key_d;

	void RotateAroundPoint(Vector3 vCenter, float angle, float X, float Y, float Z);
	void RotateView(float angle, float x, float y, float z);
	void StrafeCamera(float);
	void MoveCamera(float);
	
	void CheckForMovement();
	void Update();


	void SetDist(float v) { fDist = v; }
	void SetSpeed(float trans_sp, float rot_sp) { trans_kSpeed = trans_sp;rot_kSpeed=rot_sp; }
	SimulationSpace worldSpace;
	void SetWorldSpace(SimulationSpace worldSpace);

public:

	float	trans_kSpeed;
	float	rot_kSpeed;


	Vector3 m_vPosition;			  // where the camera is located		
	Vector3 m_vView;				 //  where the camera looks at	
	Vector3 m_vUpVector;		
	Vector3 UpVector() {	return m_vUpVector;		}
	Vector3 GetCameraView() { return m_vView; }
	Vector3 GetCameraPosition() { return m_vPosition; }


	Vector3 m_vStrafe;
	//distance from camera to view plane 
	float fDist;
	//visible angle , in Degree
	float fOv;  
	int 	ScreenWidth, ScreenHeight;		


};

#endif