#include "geometry.h"


void  RotateX(cfloat3& a, float b) {
	float tmp[9]={1,0,0,    0,cos(b),-sin(b),    0,sin(b),cos(b)};
	cmat3 tmpmat;
	tmpmat.Set(tmp);
	mvprod(tmpmat, a, a);
}

void  RotateY(cfloat3& a, float b) {
	float tmp[9] ={cos(b),0,sin(b),    0,1,0,    -sin(b),0,cos(b)};
	cmat3 tmpmat;
	tmpmat.Set(tmp);
	mvprod(tmpmat, a, a);
}

void  RotateZ(cfloat3& a, float b) {
	float tmp[9] ={cos(b),-sin(b),0,    sin(b),cos(b),0,   0,0,1};
	cmat3 tmpmat;
	tmpmat.Set(tmp);
	mvprod(tmpmat, a, a);
}

void  RotateXYZ(cfloat3& a, cfloat3& xyz) {
	cfloat3 tmp = a; //backup
	RotateX(tmp, xyz.x);
	RotateY(tmp, xyz.y);
	RotateZ(tmp, xyz.z);
	a = tmp;
}



/*-------------------------------------


			OPENGL UTILITY


---------------------------------------*/

const cmat4  IDENTITY_MAT = {{1,0,0,0,
0,1,0,0,     0,0,1,0,     0,0,0,1}};

float  cotangent(float ang){
	return 1.0/tan(ang);
}

float  deg2rad(float deg){
	return deg*cPI/180.0;
}

float  rad2deg(float rad){
	return rad*cPI/180.0;
}

void  RotateAboutX(cmat4& m,float ang){
	cmat4 rotation = IDENTITY_MAT;
	float sine = sin(ang);
	float cosine = cos(ang);

	rotation[1][1] = cosine;
	rotation[1][2] = -sine;
	rotation[2][1] = sine;
	rotation[2][2] = cosine;

	m = m * rotation; //column major, right prod = left prod in row major
}

void  RotateAboutY(cmat4& m, float ang) {
	cmat4 rotation = IDENTITY_MAT;
	float sine = sin(ang);
	float cosine = cos(ang);

	rotation[0][0] = cosine;
	rotation[0][2] = -sine;
	rotation[2][0] = sine;
	rotation[2][2] = cosine;

	m = m * rotation;
}

void  RotateAboutZ(cmat4& m, float ang) {
	cmat4 rotation = IDENTITY_MAT;
	float sine = sin(ang);
	float cosine = cos(ang);

	rotation[0][0] = cosine;
	rotation[0][1] = -sine;
	rotation[1][0] = sine;
	rotation[1][1] = cosine;

	m = m * rotation;
}

void  ScaleMatrix(cmat4& m, cfloat3 x){
	cmat4 scale = IDENTITY_MAT;
	scale[0][0] = x.x;
	scale[1][1] = x.y;
	scale[2][2] = x.z;
	m = m*scale;
}

void  TranslateMatrix(cmat4& m, cfloat3 x){
	cmat4 trans = IDENTITY_MAT;
	trans[3][0] = x.x;
	trans[3][1] = x.y;
	trans[3][2] = x.z;
	m = m*trans;
}

cmat4 CreateProjectionMatrix(float fovy, float aspect_ratio, float near_plane, float far_plane){
	cmat4 out = {{0}};
	const float 
		y_scale = cotangent(deg2rad(fovy/2)),
		x_scale = y_scale / aspect_ratio,
		frustum_length = far_plane - near_plane;
	out[0][0] = x_scale;
	out[1][1] = y_scale;
	out[2][2] = -((far_plane + near_plane)/frustum_length);
	out[2][3] = -1;
	out[3][2] = -((2*near_plane * far_plane)/frustum_length);
	
	return out;
}

float  angle(cfloat3& a,cfloat3& b){
	return acos( dot(a,b) / sqrt(dot(a,a)) /sqrt(dot(b,b)) );
}

