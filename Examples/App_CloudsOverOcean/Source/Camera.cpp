#include "Camera.h"

CCamera::CCamera(void)
{
	key_up=false; 
	key_down=false;
	key_left=false;
	key_right=false;
	key_w=false;
	key_s=false;
	key_a=false;
	key_d=false;

}
void CCamera::Init()
{
	SetSpeed((worldSpace.x_max-worldSpace.x_min)/100.0,M_PI/360);

	ScreenWidth=VIEWWIDTH;
	ScreenHeight=VIEWHEIGHT;
	Vector3 worldCenter=worldSpace.GetCenter();

	SetDist(FOCAL_LENGTH);  //focal length

	//horizontal view
	Vector3 vPosition=Vector3(worldCenter.x,worldCenter.y,worldSpace.z_max);		
	Vector3 vView=Vector3(worldCenter.x,worldCenter.y,worldCenter.z);	

	////top-down
	//Vector3 vPosition=Vector3(worldCenter.x,worldSpace.y_max,worldCenter.z);		
	//Vector3 vView=Vector3(worldCenter.x,worldCenter.y,worldCenter.z-1);	

    
	vView=vPosition+Normalize(vView-vPosition)*fDist;
	Vector3 vUpVector= Vector3(0.0, 1.0, 0.0);	

	PositionCamera(vPosition, vView,vUpVector);


}

void CCamera::PositionCamera(Vector3 vPosition	,Vector3 vView,Vector3 vUpVector)
{
	m_vPosition = vPosition;					
	m_vView     = vView;						
	m_vUpVector = vUpVector;					
}


void CCamera::RotateAroundPoint(Vector3 vCenter, float angle, float x, float y, float z)
{
	Vector3 vNewPosition;			

	Vector3 vPos = m_vPosition - vCenter;
	
	float cosTheta = (float)cos(angle);
	float sinTheta = (float)sin(angle);

	vNewPosition.x  = (cosTheta + (1 - cosTheta) * x * x)		* vPos.x;
	vNewPosition.x += ((1 - cosTheta) * x * y - z * sinTheta)	* vPos.y;
	vNewPosition.x += ((1 - cosTheta) * x * z + y * sinTheta)	* vPos.z;
	
	vNewPosition.y  = ((1 - cosTheta) * x * y + z * sinTheta)	* vPos.x;
	vNewPosition.y += (cosTheta + (1 - cosTheta) * y * y)		* vPos.y;
	vNewPosition.y += ((1 - cosTheta) * y * z - x * sinTheta)	* vPos.z;
	
	vNewPosition.z  = ((1 - cosTheta) * x * z - y * sinTheta)	* vPos.x;
	vNewPosition.z += ((1 - cosTheta) * y * z + x * sinTheta)	* vPos.y;
	vNewPosition.z += (cosTheta + (1 - cosTheta) * z * z)		* vPos.z;

	m_vPosition = vCenter + vNewPosition;

    vPos = m_vView - vCenter;

	vNewPosition.x  = (cosTheta + (1 - cosTheta) * x * x)		* vPos.x;
	vNewPosition.x += ((1 - cosTheta) * x * y - z * sinTheta)	* vPos.y;
	vNewPosition.x += ((1 - cosTheta) * x * z + y * sinTheta)	* vPos.z;

	vNewPosition.y  = ((1 - cosTheta) * x * y + z * sinTheta)	* vPos.x;
	vNewPosition.y += (cosTheta + (1 - cosTheta) * y * y)		* vPos.y;
	vNewPosition.y += ((1 - cosTheta) * y * z - x * sinTheta)	* vPos.z;

	vNewPosition.z  = ((1 - cosTheta) * x * z - y * sinTheta)	* vPos.x;
	vNewPosition.z += ((1 - cosTheta) * y * z + x * sinTheta)	* vPos.y;
	vNewPosition.z += (cosTheta + (1 - cosTheta) * z * z)		* vPos.z;

	m_vView = vCenter + vNewPosition;
}

void CCamera::RotateView(float angle, float x, float y, float z)
{
	Vector3 vNewView;

	// Get the view vector (The direction we are facing)
	Vector3 vView = m_vView - m_vPosition;		

	// Calculate the sine and cosine of the angle once
	float cosTheta = (float)cos(angle);
	float sinTheta = (float)sin(angle);

	// Find the new x position for the new rotated point
	vNewView.x  = (cosTheta + (1 - cosTheta) * x * x)		* vView.x;
	vNewView.x += ((1 - cosTheta) * x * y - z * sinTheta)	* vView.y;
	vNewView.x += ((1 - cosTheta) * x * z + y * sinTheta)	* vView.z;

	// Find the new y position for the new rotated point
	vNewView.y  = ((1 - cosTheta) * x * y + z * sinTheta)	* vView.x;
	vNewView.y += (cosTheta + (1 - cosTheta) * y * y)		* vView.y;
	vNewView.y += ((1 - cosTheta) * y * z - x * sinTheta)	* vView.z;

	// Find the new z position for the new rotated point
	vNewView.z  = ((1 - cosTheta) * x * z - y * sinTheta)	* vView.x;
	vNewView.z += ((1 - cosTheta) * y * z + x * sinTheta)	* vView.y;
	vNewView.z += (cosTheta + (1 - cosTheta) * z * z)		* vView.z;

	// Now we just add the newly rotated vector to our position to set
	// our new rotated view of our camera.
	m_vView = m_vPosition + vNewView;
}

void CCamera::StrafeCamera(float speed)
{	
	// Add the strafe vector to our position
	m_vPosition.x += m_vStrafe.x * speed;
	m_vPosition.y += m_vStrafe.y * speed;

	// Add the strafe vector to our view
	m_vView.x += m_vStrafe.x * speed;
	m_vView.y+= m_vStrafe.y * speed;
}


void CCamera::MoveCamera(float speed)
{
	// Get the current view vector (the direction we are looking)
	Vector3 vVector = m_vView - m_vPosition;
	vVector = Normalize(vVector);

	m_vPosition.x += vVector.x * speed;
	m_vPosition.y += vVector.y * speed;
	m_vPosition.z += vVector.z * speed;
	m_vView.x += vVector.x * speed;
	m_vView.y += vVector.y * speed;
	m_vView.z += vVector.z * speed;
}



void CCamera::CheckForMovement()
{	
	float speed = trans_kSpeed;
	if (key_up)	
	{
		MoveCamera(speed);	
	   key_up=false;

	}
	if (key_down) 
	{
		MoveCamera(-speed);
		key_down=false;
	}
	if (key_left) 
	{
		StrafeCamera(speed);
		key_left=false;
	}
	if (key_right) 
	{
		StrafeCamera(-speed);
		key_right=false;
	}
	
	Vector3 worldCenter=Vector3((worldSpace.x_max+worldSpace.x_min)/2,(worldSpace.y_max+worldSpace.y_min)/2,(worldSpace.z_max+worldSpace.z_min)/2);
	speed = rot_kSpeed;
	if(key_w) 
	{
		RotateAroundPoint(worldCenter,speed,  1, 0,0);
		key_w=false;
	}
	if(key_s) 
	{
		RotateAroundPoint(worldCenter,-speed,  1, 0,0);
		key_s=false;

	}
	if(key_a)   
	{
		RotateAroundPoint(worldCenter,speed, 0, 1, 0);
		key_a=false;
	}
	if(key_d)  
	{
		RotateAroundPoint(worldCenter,-speed, 0, 1, 0);
		key_d=false;

	}

}

void CCamera::Update() 
{
	Vector3 vCross = Cross(m_vView - m_vPosition, m_vUpVector);
	m_vStrafe = Normalize(vCross);
	CheckForMovement();

}

CCamera::~CCamera()
{

}

void CCamera::SetWorldSpace( SimulationSpace worldSpace )
{
	this->worldSpace=worldSpace;

}






