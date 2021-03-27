#include "Pixel.h"
#include "POINT.h"
#include <algorithm>
#include "Mesh.h"
Pixel::Pixel()
{
	cout << "creat an pixel object" << endl;
}

void Pixel::Initial()
{
	skycount = 0;
	cloudPixelCount = 0;
	seg_thresh = 0.479;
	horizontalLine = 280;
	boundaryPixel_number = 0;
	pixelTypeList = NULL;
	boundaryPixelIndexList = NULL;
}
int * Pixel::GetPixelTypeList()
{
	return this -> pixelTypeList;
}

int * Pixel::GetBoundaryList()
{
	return this ->boundaryPixelIndexList;
}

int Pixel::GetBoundaryPixel_num()
{
	return this -> boundaryPixel_number;
}

void Pixel::CreatePixelType(Image image, Tool tool)
{
	tool.PrintRunnngIfo("Create Pixel Type");

	pixelTypeList = new int[image.GetImg_height()*image.GetImg_width()];
	if (pixelTypeList == NULL)
	{
		cout << "allocate memory for pixel type failed!\n";
		exit(1);
	}
	//for the type of pixel, -1---boundary, 1---cloud,0---sky,2--ground,
	for (int i = horizontalLine; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			pixelTypeList[i*image.GetImg_width() + j] = 2;
		}

	int  height = min(horizontalLine, image.GetImg_height());
	//将图像分割，分离出地面，天空和云
	for (int i = 0; i<height; i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{

			Color3 color = image.GetImg_mat_cor()[i*image.GetImg_width() + j];
			if ((color.maxPart() - color.minPart()) / color.maxPart()<seg_thresh)//seg_thresh应该是设置的阈值，分离出云和天空
			{
				pixelTypeList[i*image.GetImg_width() + j] = 1;
				cloudPixelCount++;
			}
			else
			{
				pixelTypeList[i*image.GetImg_width() + j] = 0;
				skycount++;

			}

		}

	bool* CloudBoudaryPixelList = new bool[image.GetImg_height()*image.GetImg_width()];
	if (CloudBoudaryPixelList == NULL)
	{
		cout << "allocate memory for cloud boundary pixel failed!\n";
		exit(1);
	}

	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixelTypeList[i*image.GetImg_width() + j] == 1)
			{
				if (i - 1>0 && pixelTypeList[(i - 1)*image.GetImg_width() + j] != 1)
				{
					CloudBoudaryPixelList[i*image.GetImg_width() + j] = true;
					continue;
				}
				if (i + 1<image.GetImg_height() && pixelTypeList[(i + 1)*image.GetImg_width() + j] != 1)
				{
					CloudBoudaryPixelList[i*image.GetImg_width() + j] = true;
					continue;
				}
				if (j - 1>0 && pixelTypeList[(i)*image.GetImg_width() + j - 1] != 1)
				{
					CloudBoudaryPixelList[i*image.GetImg_width() + j] = true;
					continue;
				}
				if (j + 1<image.GetImg_width() && pixelTypeList[(i)*image.GetImg_width() + j + 1] != 1)
				{
					CloudBoudaryPixelList[i*image.GetImg_width() + j] = true;
					continue;
				}

				CloudBoudaryPixelList[i*image.GetImg_width() + j] = false;

			}

		}
	//在云的点周围，通过云的四周判断，确定出云的边界

	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixelTypeList[i*image.GetImg_width() + j] == 1 && CloudBoudaryPixelList[i*image.GetImg_width() + j])
			{
				pixelTypeList[i*image.GetImg_width() + j] = -1;
			}
		}

	delete[] CloudBoudaryPixelList;
	
	/*ofstream out("../output/pixelifo_test.txt");
	out << 414 << endl;
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			out << pixelTypeList[i*image.GetImg_width() + j] << " ";

		}
	out << endl;*/
	
	//cv::Mat img(image.GetImg_height(), image.GetImg_width(),CV_8UC1);
	//for (int i = 0; i < image.GetImg_height(); ++i) {
	//	for (int j = 0; j < image.GetImg_width(); ++j) {
	//		img.at<uchar>(i,j) = static_cast<uchar>((pixelTypeList[i * image.GetImg_width() + j]+1)*255/3);
	//	}
	//}
	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Display window", img);
	//cv::waitKey(0);
}

void Pixel::CreatePerfectBoundary(Image image, Mesh& mesh)
{
	if (pixelTypeList == NULL)
		return;


	int* cloudMask = new int[image.GetImg_width() * image.GetImg_height()];

	//---------------
	//内部采点
	int count = 0;
	//---------------

	for (int i = 0; i < image.GetImg_height(); i++)
		for (int j = 0; j < image.GetImg_width(); j++)
		{
			cloudMask[i * image.GetImg_width() + j] = 0;

			if (pixelTypeList[i * image.GetImg_width() + j] == 1) //cloud
			{
				cloudMask[i * image.GetImg_width() + j] = 255;
			}
		}

	IplImage* g_gray = NULL;
	int g_thresh = 120; //设置阈值
	CvMemStorage* g_storage = NULL;//内存存储器是一个可用来存储诸如序列，轮廓，图形,子划分等动态增长数据结构的底层结构。它是由一系列以同等大小的内存块构成，呈列表型 。
	if (g_storage == NULL)
	{
		CvSize  cs;//CvSize，OpenCV的基本数据类型之一。表示矩阵框大小，以像素为精度。与CvPoint结构类似，但数据成员是integer类型的width和height。
		cs.width = image.GetImg_width();
		cs.height = image.GetImg_height();
		g_gray = cvCreateImage(cs, 8, 1);//初始化一张灰度图像
		g_storage = cvCreateMemStorage(0);//初始化内存器大小，为64k
	}
	else
		cvClearMemStorage(g_storage);

	for (int i = 0; i < image.GetImg_height(); i++)
		for (int j = 0; j < image.GetImg_width(); j++)
		{
			cvSetReal2D(g_gray, i, j, cloudMask[i * image.GetImg_width() + j]);

		}
	delete[] cloudMask;

	//SaveImage("../output/gray.jpg", g_gray);

	//------------manually modify the image---------------- 手动修改图片
	cout << "If necessary!  modify the gray image!" << endl;

	//--------------------------------------------------------------------------------------------
	//cv::Mat gray = cv::imread("../output/gray.jpg");
	//if (gray.type() != 0) {
	//	cv::cvtColor(gray, gray, cv::COLOR_RGB2GRAY);
	//}
	//double max_contour_area = 0;
	//int max_contour_area_index = 0;
	//vector<vector<cv::Point>> contours;
	//vector<cv::Vec4i> hierarchy;
	//cv::findContours(gray, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	//// find the largest area countour of image
	//for (int i = 0; i < contours.size(); i++) {
	//	double area = cv::contourArea(contours[i]);
	//	if (area >= max_contour_area) {
	//		max_contour_area = area;
	//		max_contour_area_index = i;
	//	}
	//}
	//mesh.contour = contours[max_contour_area_index];
	//for (int i = 0; i < mesh.contour.size(); i++) {
	//	mesh.contour[i].y = image.img_height - mesh.contour[i].y;
	//}

	cv::Point first_contour;
	vector<bool> is_used(image.GetImg_height() * image.GetImg_width(), false);
	for (int i = 0; i < image.GetImg_height(); i++) {
		bool flag = false;
		for (int j = 0; j < image.GetImg_width(); j++)
		{
			if (pixelTypeList[i * image.GetImg_width() + j] == -1)
			{
				first_contour.x = j;
				first_contour.y = image.GetImg_height() - i;
				flag = true;
				break;
			}
		}
		if (flag)
			break;
	}

	int max_dist = 5;
	vector<int> i_pos;
	vector<int> j_pos;
	for (int m = 1; m <= max_dist; m++) {
		int side = 2 * m + 1;
		for (int i = 0; i < side; i++) {
			i_pos.push_back(-m);
			j_pos.push_back(-m + i);
		}
		for (int i = 1; i < side - 1; i++) {
			i_pos.push_back(-m + i);
			j_pos.push_back(m);
		}
		for (int i = 0; i < side; i++) {
			i_pos.push_back(m);
			j_pos.push_back(m - i);
		}
		for (int i = 1; i < side - 1; i++) {
			i_pos.push_back(m - i);
			j_pos.push_back(-m);
		}
	}

	int k = 0;
	int search_size = i_pos.size();
	mesh.contour.push_back(first_contour);
	is_used[(image.GetImg_height() - mesh.contour[0].y) * image.GetImg_width() + mesh.contour[0].x] = true;
	while (1) {
		int i = image.GetImg_height() - mesh.contour[k].y;
		int j = mesh.contour[k].x;
		bool flag = false;
		for (int t = 0; t < search_size; t++) {
			if ((i + i_pos[t]) >= 0 && (i + i_pos[t]) < image.GetImg_height() &&
				(j + j_pos[t]) >= 0 && (j + j_pos[t]) < image.GetImg_width() &&
				pixelTypeList[(i + i_pos[t]) * image.GetImg_width() + j + j_pos[t]] == -1 &&
				is_used[(i + i_pos[t]) * image.GetImg_width() + j + j_pos[t]] == false) {

				mesh.contour.push_back(cv::Point(j + j_pos[t], image.GetImg_height() - i - i_pos[t]));
				is_used[(i + i_pos[t]) * image.GetImg_width() + j + j_pos[t]] = true;
				k++;
				flag = true;
				break;
			}
		}
		if (!flag)
			break;
	}




	//vector<cv::Point> contour_point;
	////vector<bool> is_used(image.GetImg_height()* image.GetImg_width(), false);
	//for (int i = 0; i < image.GetImg_height(); i++) {
	//	for (int j = 0; j < image.GetImg_width(); j++)
	//	{
	//		if (pixelTypeList[i * image.GetImg_width() + j] == -1)
	//		{
	//			contour_point.push_back(cv::Point(j, image.GetImg_height() - i));
	//		}
	//	}
	//}
	//mesh.contour.push_back(contour_point[0]);
	//int k = 0;
	//int i_pos[] = { 0,-1,1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8,8,-9,9,-10,10 };
	//int j_pos[] = { 0,-1,1,-2,2,-3,3,-4,4,-5,5,-6,6,-7,7,-8,8,-9,9,-10,10 };
	//int ran[] = { 3,5,7,9,11,13,15,17,19,21 };
	//is_used[(image.GetImg_height() - mesh.contour[0].y) * image.GetImg_width() + mesh.contour[0].x] = true;
	//while (1) {
	//	int i = image.GetImg_height() - mesh.contour[k].y;
	//	int j = mesh.contour[k].x;
	//	bool flag = false;
	//	for (int m = 0; m < 10; m++) {
	//		for (int ti = 0; ti < ran[m]; ti++) {
	//			for (int tj = 0; tj < ran[m]; tj++) {
	//				if ((i + i_pos[ti]) >= 0 && (i + i_pos[ti]) < image.GetImg_height() &&
	//					(j + j_pos[tj]) >= 0 && (j + j_pos[tj]) < image.GetImg_width() &&
	//					pixelTypeList[(i + i_pos[ti]) * image.GetImg_width() + j + j_pos[tj]] == -1 &&
	//					is_used[(i + i_pos[ti]) * image.GetImg_width() + j + j_pos[tj]] == false) {

	//					mesh.contour.push_back(cv::Point(j + j_pos[tj], image.GetImg_height() - i - i_pos[ti]));
	//					is_used[(i + i_pos[ti]) * image.GetImg_width() + j + j_pos[tj]] = true;
	//					k++;
	//					flag = true;
	//					break;
	//				}
	//			}
	//			if (flag)
	//				break;
	//		}
	//		if (flag)
	//			break;
	//	}
	//	if (flag)
	//		continue;

	//	break;
	//}
	//assert((k + 1) == contour_point.size());

}

void Pixel::SetPixelTypeList(int i, int value)
{
	pixelTypeList[i] = value;
}
