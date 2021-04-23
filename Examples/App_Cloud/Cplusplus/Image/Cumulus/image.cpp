#include "Image.h"

Image::Image()
{
	cout << "creat an image object" << endl;
}
void Image::Initial(void)
{
	this->pImg = NULL;
	img_width = 0;
	img_height = 0;
	img_mat = NULL;
	img_mat_cor = NULL;
}

void Image::ReadImage(const char* imgfile, Tool tool)
{
	tool.PrintRunnngIfo("Reading image");
	//read image data to img_mat_cor;
	pImg = cvLoadImage(imgfile, 1); //从指定文件中读入图像，返回读入图像的指针，1代表彩色图
	//cvSaveImage("E:/代码/mesh_x_new/output/out.png", pImg);
	img_width = pImg->width;
	img_height = pImg->height;
	img_maxWH = img_width>img_height ? img_width : img_height;
	img_mat_cor = new Color3[img_width*img_height];//初始化色彩矩阵，数组中每一个元素都是color3的示例对象

												   //allocate memory for img_mat_cor
	if (img_mat_cor == NULL)
	{
		cout << "img_mat_cor allocation failed!\n";
		exit(1);
	}

	for (int i = 0; i<img_height; i++)
		for (int j = 0; j<img_width; j++)
		{
			CvScalar cs;//CvScalar是一个可以用来存放4个double数值的数组。
			cs = cvGet2D(pImg, i, j);
			img_mat_cor[i*img_width + j].R = cs.val[2] / 255;
			img_mat_cor[i*img_width + j].G = cs.val[1] / 255;
			img_mat_cor[i*img_width + j].B = cs.val[0] / 255;//为什么要除以255
		}

	////方法一：使用cvGet2D()函数间接访问，若是彩色图，就是3通道图像，获取的就是每一个像素点的BGR值，然后分别获取B值，G值和R值
	//    CvScalar s=cvGet2D(img,i,j); //i代表y轴，即height；j代表x轴，即width。
	//    printf("B=%f, G=%f, R=%f\n",s.val[0],s.val[1],s.val[2]);    //注意是BGR顺序


	//read image data to img_mat;
	pImg = cvLoadImage(imgfile, 0);//强制转化读取图像为灰度图

	img_mat = new float[img_width*img_height];//初始化灰度图像数组大小

											  //allocate memory for img_mat
	if (img_mat == NULL)
	{
		cout << "img_mat allocation failed!\n";
		exit(1);
	}

	//smooth the image data,remove noise
	int smoothCount = 1;
	while (smoothCount--)
		cvSmooth(pImg, pImg, CV_GAUSSIAN, 3, 0, 3, 0);
	//将图像进行高斯平滑处理，消除噪声
	//normalize gray value

	for (int i = 0; i<img_height; i++)
		for (int j = 0; j<img_width; j++)
		{
			img_mat[i*img_width + j] = cvGetReal2D(pImg, i, j) / 255;//i是行参数，j是列参数

		}
}

int Image::GetImg_width()
{
	return this->img_width;
}

int Image::GetImg_height()
{
	return this->img_height;
}

int Image::GetImg_maxWH()
{
	return this -> img_maxWH;
}

Color3* Image::GetImg_mat_cor()
{
	return this->img_mat_cor;
}

float * Image::GetImg_mat()
{
	return this -> img_mat;
}
