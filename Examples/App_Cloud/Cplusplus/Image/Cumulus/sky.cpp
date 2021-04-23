#include "sky.h"

Sky::Sky()
{
	cout << "creat a sky object" << endl;
}

void Sky::Initial()
{
	img_sky = NULL;
	img_grey_sky = NULL;
}

void Sky::CreateSkyPossion(Image image, Pixel pixel)//利用插值法构造天空场景，保存天空灰度的值用于高度场计算公式
{
	int*  isSkyPixelList = new int[image.GetImg_width()*image.GetImg_height()];
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			isSkyPixelList[i*image.GetImg_width() + j] = pixel.GetPixelTypeList()[i*image.GetImg_width() + j];
		}


	int neighbor_radius = 5;
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			bool isFind = false;
			for (int k = -neighbor_radius; k<neighbor_radius && !isFind; k++)
				for (int n = -neighbor_radius; n<neighbor_radius && !isFind; n++)
				{
					int idx = j + n;
					int idy = i + k;
					if (idx>0 && idx<image.GetImg_width() &&idy>0 && idy<image.GetImg_height())
					{
						if (pixel.GetPixelTypeList()[idy*image.GetImg_width() + idx] != 0)
						{
							isFind = true;
						}
					}

				}

			if (isFind)
			{
				isSkyPixelList[i*image.GetImg_width() + j] = 0;
			}

			else
			{
				isSkyPixelList[i*image.GetImg_width() + j] = 1;
			}


		}


	img_sky = new Color3[image.GetImg_width()*image.GetImg_height()];
	Color3*  tempCor = new Color3[image.GetImg_width()*image.GetImg_height()];

	Color3 avgSky(0, 0, 0);		//average sky color		
	int skycount = 0;
	int cloudPixelCount = 0;
	for (int i = 0; i<image.GetImg_width()*image.GetImg_height(); i++)
	{
		if (isSkyPixelList[i] == 1)
		{
			img_sky[i] = image.GetImg_mat_cor()[i];
			avgSky += image.GetImg_mat_cor()[i];
			skycount++;
		}


	}
	avgSky = avgSky*(1.0 / skycount);
	avgSky = Color3(0.02, 0.02, 0.02);

	//set colors of cloud  pixel to avgsky
	cloudPixelCount = 0;
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (isSkyPixelList[i*image.GetImg_width() + j] == 1)
			{
				img_sky[i*image.GetImg_width() + j] = avgSky;
				cloudPixelCount++;

			}
		}

	int Max_Iter = 5000;

	float Error = 999999999;

	neighbor_radius = 3;

	//特别花时间！！
	for (int iter = 0; iter<Max_Iter&&Error>1.0e-20; iter++)
	{
		for (int i = 0; i<image.GetImg_width()*image.GetImg_height(); i++)
		{
			tempCor[i] = Color3(0, 0, 0);
		}

		for (int i = 0; i<image.GetImg_height(); i++)
			for (int j = 0; j<image.GetImg_width(); j++)
			{
				if (isSkyPixelList[i*image.GetImg_width() + j] == 0 && i >= 0 && i<image.GetImg_height()&&j >= 0 && j<image.GetImg_width())
				{
					int neighorcount = 0;
					float weightSum = 0.0;
					for (int k = -neighbor_radius; k<neighbor_radius; k++)
						for (int n = -neighbor_radius; n<neighbor_radius; n++)
						{
							if (n*n + k*k != 0)
							{
								int idx = j + n;
								int idy = i + k;
								if (idx>0 && idx<image.GetImg_width()&&idy>0 && idy<image.GetImg_height())
								{
									if (isSkyPixelList[idy*image.GetImg_width() + idx] == 1)
										tempCor[i*image.GetImg_width() + j] += image.GetImg_mat_cor()[idy*image.GetImg_width() + idx];
									else
										tempCor[i*image.GetImg_width() + j] += img_sky[idy*image.GetImg_width() + idx];

									neighorcount++;

								}
							}
						}

					Color3 temp = tempCor[i*image.GetImg_width() + j] * (1.0 / neighorcount);
					tempCor[i*image.GetImg_width() + j] = tempCor[i*image.GetImg_width() + j] * (1.0 / neighorcount);


				}

			}

		float curError = 0;
		for (int i = 0; i<image.GetImg_width()*image.GetImg_height(); i++)
		{
			if (isSkyPixelList[i] == 0)
			{
				Color3 difCor = tempCor[i] - img_sky[i];
				curError += difCor.R*difCor.R + difCor.G*difCor.G + difCor.B*difCor.B;

			}
		}

		curError = sqrt(curError / cloudPixelCount);


		//cout<<"IterCount:" <<iter<<"curError: "<<curError<<endl;
		for (int i = 0; i<image.GetImg_width()*image.GetImg_height(); i++)
		{
			if (isSkyPixelList[i] == 1)
			{
				img_sky[i] = image.GetImg_mat_cor()[i];

			}
			else
				img_sky[i] = tempCor[i];

		}
		Error = curError;

	}


	//convert the colorful  sky to grey image
	img_grey_sky = new float[image.GetImg_width()*image.GetImg_height()];

	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			float R = img_sky[i*image.GetImg_width() + j].R;
			float G = img_sky[i*image.GetImg_width() + j].G;
			float B = img_sky[i*image.GetImg_width() + j].B;
			img_grey_sky[i*image.GetImg_width() + j] = 0.2989 * R + 0.5870 * G + 0.1140 * B;//0.2989 * R + 0.5870 * G + 0.1140 * B ;

		}


	float max_sky_dif_cloud = -99999;

	int nSingularPixel = 0;
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1)
			{
				if (img_grey_sky[i*image.GetImg_width() + j] - image.GetImg_mat()[i*image.GetImg_width() + j]>0)
				{
					nSingularPixel++;

				}

			}
		}
	cout << "The number of pixels whose sky intensities being greater than image intensities:   " << nSingularPixel << endl;

	delete[] isSkyPixelList;
}

float* Sky::GetImg_grey_sky()
{
	return this->img_grey_sky;
}
