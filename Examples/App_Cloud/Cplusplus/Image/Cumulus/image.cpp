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
	pImg = cvLoadImage(imgfile, 1); //��ָ���ļ��ж���ͼ�񣬷��ض���ͼ���ָ�룬1�����ɫͼ
	//cvSaveImage("E:/����/mesh_x_new/output/out.png", pImg);
	img_width = pImg->width;
	img_height = pImg->height;
	img_maxWH = img_width>img_height ? img_width : img_height;
	img_mat_cor = new Color3[img_width*img_height];//��ʼ��ɫ�ʾ���������ÿһ��Ԫ�ض���color3��ʾ������

												   //allocate memory for img_mat_cor
	if (img_mat_cor == NULL)
	{
		cout << "img_mat_cor allocation failed!\n";
		exit(1);
	}

	for (int i = 0; i<img_height; i++)
		for (int j = 0; j<img_width; j++)
		{
			CvScalar cs;//CvScalar��һ�������������4��double��ֵ�����顣
			cs = cvGet2D(pImg, i, j);
			img_mat_cor[i*img_width + j].R = cs.val[2] / 255;
			img_mat_cor[i*img_width + j].G = cs.val[1] / 255;
			img_mat_cor[i*img_width + j].B = cs.val[0] / 255;//ΪʲôҪ����255
		}

	////����һ��ʹ��cvGet2D()������ӷ��ʣ����ǲ�ɫͼ������3ͨ��ͼ�񣬻�ȡ�ľ���ÿһ�����ص��BGRֵ��Ȼ��ֱ��ȡBֵ��Gֵ��Rֵ
	//    CvScalar s=cvGet2D(img,i,j); //i����y�ᣬ��height��j����x�ᣬ��width��
	//    printf("B=%f, G=%f, R=%f\n",s.val[0],s.val[1],s.val[2]);    //ע����BGR˳��


	//read image data to img_mat;
	pImg = cvLoadImage(imgfile, 0);//ǿ��ת����ȡͼ��Ϊ�Ҷ�ͼ

	img_mat = new float[img_width*img_height];//��ʼ���Ҷ�ͼ�������С

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
	//��ͼ����и�˹ƽ��������������
	//normalize gray value

	for (int i = 0; i<img_height; i++)
		for (int j = 0; j<img_width; j++)
		{
			img_mat[i*img_width + j] = cvGetReal2D(pImg, i, j) / 255;//i���в�����j���в���

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
