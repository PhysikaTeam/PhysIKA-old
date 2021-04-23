#pragma once
#include "Tool.h"
#include "Image.h"
//�����໥����ã�Ҫ����һ�����ͷ�ļ���������һ���࣡����
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
	int* boundaryPixelIndexList;//��������ʱʹ��
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