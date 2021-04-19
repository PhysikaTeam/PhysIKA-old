#include "Sun.h"
#include <algorithm>
Sun::Sun()
{
	cout << "creat an sun object" << endl;
}

void Sun::Initial()
{
	thetaSun = 0.0;
	phiSun = 0.0;
	focalLength = 638.88;
	brighter_ratio = 0.1;
}

void Sun::CreateSunColor(Pixel pixel, Image image)
{
	struct PixelIFO
	{
		int row;
		int col;
		float  gray;
		Color3  color;

		bool operator <(const PixelIFO& other)
		{
			return 	this->gray > other.gray;
		}

	};

	vector<PixelIFO> cloudIfoList;

	for (int i = 0; i < image.GetImg_height(); i++)
		for (int j = 0; j < image.GetImg_width(); j++)
		{

			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1) //cloud
			{
				PixelIFO curPixel;
				curPixel.row = i;
				curPixel.col = j;
				curPixel.gray = image.GetImg_mat()[i*image.GetImg_width() + j];
				curPixel.color = image.GetImg_mat_cor()[i*image.GetImg_width() + j];
				cloudIfoList.push_back(curPixel);//在cloudIfo后加一个结构包括行，列和灰度值和rgb色彩值

			}

		}

	std::sort(cloudIfoList.begin(), cloudIfoList.end());//排序



	int  brighter_Number = cloudIfoList.size()*brighter_ratio;

	Color3 avg_cor = Color3(0, 0, 0);
	for (int i = 0; i < brighter_Number; i++)
	{
		int idx = cloudIfoList[i].col;
		int idy = cloudIfoList[i].row;
		avg_cor += image.GetImg_mat_cor()[idy*image.GetImg_width() + idx];

	}

	avg_cor.R /= brighter_Number;
	avg_cor.G /= brighter_Number;
	avg_cor.B /= brighter_Number;
	//ofstream out("../output/suncolor_test.txt");
	//out << "Sun Color: " << avg_cor.R << " " << avg_cor.G << " " << avg_cor.B << endl;


	sun_color = avg_cor;//计算太阳色彩
}

void Sun::SetThetaSun(float theta)
{
	this->thetaSun = theta;
}

void Sun::SetPhiSun(float phi)
{
	this->phiSun = phi;
}

void Sun::CreateSun(float theta, float phi)
{
	thetaSunUV = theta;
	phiSunUV = phi;

	SunVecUV.x = sin(theta)*cos(phi);
	SunVecUV.z = sin(theta)*sin(phi);
	SunVecUV.y = cos(theta);
}

Vector3 Sun::GetSunVecUV()
{
	return this ->SunVecUV;
}
