#pragma once
#include "Color.h"
#include "Pixel.h"
#include "Image.h"
class Sun {
public:
	Color3  sun_color;
	float brighter_ratio;
	float  focalLength;
	float thetaSun;//zenith angle of sun in world coordinate
	float phiSun; // azimuth angle of sun in world coordinate
	Vector3 SunVec; //sun direction in world coordinate,x-y-z, y-right, z-up, x points at the user.
	float thetaSunUV;//zenith angle of sun in image coordinate
	float phiSunUV; // azimuth angle of sun in image coordinate
	Vector3 SunVecUV;// sun direction in image coordinate, u-v-w, u-right, v-up, w points at the user.
	float scale_coff[3];
	Sun();
	void Initial();
	void CreateSunColor(Pixel pixel,Image image);
	void SetThetaSun(float theta);
	void SetPhiSun(float phi);
	void CreateSun(float theta, float phi);
	Vector3 GetSunVecUV();
};