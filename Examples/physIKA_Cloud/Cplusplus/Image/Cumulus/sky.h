#pragma once
#include "Color.h"
#include "Image.h"
#include "Pixel.h"
class Sky {
public:
	Color3* img_sky;
	float*  img_grey_sky;
	Sky();
	void Initial();
	void CreateSkyPossion(Image image, Pixel pixel);
	float* GetImg_grey_sky();
};