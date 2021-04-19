#pragma once
#include "global.h"
#include "POINT.h"
#include "CloudInterior.h"
#include "Image.h"
#include "Sun.h"
#include "pixel.h"
#include "sky.h"
#include "Perlin.h"
//************
#include"Mesh.h"

class Cloud {
public:
	CloudInterior  cloudIn;
	CloudVolume curCloudVolume;//云的圆柱体集合迭代产生
	vector<POINT>   path;
	Perlin* perlin;
	float*  heightField;
	float* otherHeightField;
	void Initial();
	void CreatePropagationPath(Image image, Sun sun, Pixel pixel);
	Cylinder CreateCylinder(int x_index, int y_index,Image image,Sky sky, Sun sunS);
	void BiSectionMethod(int px, int py, Cylinder& curCylinder, Image image,Sky sky, Sun sun);
	void PropagationCylinders(Image image, Sky sky, Sun sun);
	float ComputeSingleScattering(float H, int px, int py, Image image, Sky sky, Sun sun);
	float ComputeNormalHeightNeightborAvg(int px, int py, Image image, Pixel pixel,Sky sky);
	float PhaseFunction(Vector3 v1, Vector3 v2);
	void  CreateHeightFieldHalf(Image image, Pixel pixel, Sky sky);

	void SetHeightField(int i, float value);

	//---------------
	void CreateHeightFieldOtherHalf(Image image, Pixel pixel, Perlin* perlin);
	//---------------

	//---------------
	void RemoveOneLoopPixelBoudary(Image image, Pixel pixel, float* heightField);
	//---------------

	//---------------
	float InterPolateHeightField(Image image, Pixel pixel, Sky sky, float x, float y, float* heightField);

	void NormalizeCloudMesh(Mesh mesh);

	void CreateCloudMesh(Image image, Pixel pixel, Sky sky, Mesh &mesh, float* heightField);
	//---------------

	//---------------
	void ExportCloudMesh(Mesh& mesh, char* filename);
	//---------------
	
};