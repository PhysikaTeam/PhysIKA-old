#pragma once
#include "Tool.h"
#include "Image.h"
//两个类互相调用，要先在一个类的头文件中声明另一个类！！！
//#include "Mesh.h"
class Mesh;
class Pixel 
{
public:
	int*  pixelTypeList;
	float seg_thresh;
	int skycount;
	int horizontalLine;
	int cloudPixelCount;
	int* boundaryPixelIndexList;//创建网格时使用
	int  boundaryPixel_number;
	Pixel();
	void Initial();
	int* GetPixelTypeList();
	int* GetBoundaryList();
	int GetBoundaryPixel_num();
	void CreatePixelType(Image image, Tool tool);
	void CreatePerfectBoundary(Image image, Mesh &mesh);
	void SetPixelTypeList(int i, int value);
};