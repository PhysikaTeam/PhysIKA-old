#pragma once
#include "global.h"
#include "Color.h"
#include "Tool.h"
class Image {
public:
	IplImage* pImg;
	int img_width;
	int img_height;
	int img_maxWH;
	float *   img_mat;
	Color3* img_mat_cor;
	Image();
	void Initial(void);
	void ReadImage(const char* filename, Tool tool);
	int GetImg_width();
	int GetImg_height();
	int GetImg_maxWH();
	Color3* GetImg_mat_cor();
	float* GetImg_mat();

};