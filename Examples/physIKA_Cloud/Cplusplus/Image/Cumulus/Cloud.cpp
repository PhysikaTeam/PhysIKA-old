#include "Cloud.h"
#include <algorithm>
void Cloud::Initial()
{
	heightField = NULL;
	otherHeightField = NULL;
	perlin = new Perlin(4, 4, 1, 94);
}

void Cloud::CreatePropagationPath(Image image, Sun sun, Pixel pixel)
{
	path.clear();//二维点向量构成的路径


	Vector3  sunVec = sun.GetSunVecUV() * 2;//
	float k = -sunVec.x / sunVec.y;

	float b = sunVec.y - k*sunVec.x;//计算与光源垂直投影后到视平面，与够成面垂直的直线

	//  kx-y+b=0

	float*  distanceToLine = new float[image.GetImg_width()*image.GetImg_height()];
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			float cur_x = float(j) / image.GetImg_maxWH();
			float cur_y = float(image.GetImg_height() - i) / image.GetImg_maxWH();

			distanceToLine[i*image.GetImg_width() + j] = fabs(k*cur_x - cur_y + b) / sqrtf(k*k + 1);//存储图像上阳光直射线上的距离

		}

	class  PixelIfo
	{
	public:
		int x;
		int y;
		float   dis2Sun;
		PixelIfo(int x, int y, float dis)
		{
			this->x = x;
			this->y = y;
			this->dis2Sun = dis;
		}

		bool operator <(const PixelIfo& other)
		{
			return 	this->dis2Sun<other.dis2Sun;
		}
	};
	//定义像素结构信息类
	vector<PixelIfo>  pixelIflList;

	float maxDis = -MAXVAL;
	float minDis = MAXVAL;
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1)
			{
				float cur_x = float(j) / image.GetImg_width();
				float cur_y = float(image.GetImg_height() - i) / image.GetImg_maxWH();

				distanceToLine[i*image.GetImg_width() + j] = fabs(k*cur_x - cur_y + b) / sqrtf(k*k + 1);
				maxDis = max(maxDis, distanceToLine[i*image.GetImg_width() + j]);
				minDis = min(minDis, distanceToLine[i*image.GetImg_width() + j]);
				pixelIflList.push_back(PixelIfo(j, i, distanceToLine[i*image.GetImg_width() + j]));


			}

		}

	//normalize the distance
	for (int i = 0; i<pixelIflList.size(); i++)
	{

		pixelIflList[i].dis2Sun = (pixelIflList[i].dis2Sun - minDis) / (maxDis - minDis);

	}


std:sort(pixelIflList.begin(), pixelIflList.end());
	//将左边变为正常的距离并作为排序

	for (int i = 0; i<pixelIflList.size(); i++)
	{
		POINT  pt;
		pt.x = pixelIflList[i].x;
		pt.y = pixelIflList[i].y;
		path.push_back(pt);

	}
	//记录下传播的路径

}

Cylinder Cloud::CreateCylinder(int x_index, int y_index, Image image, Sky sky ,Sun sun)
{
	float H = 0.3;
	Vector3 center(1.0*x_index / image.GetImg_maxWH(), 1.0*(image.GetImg_height() - y_index) / image.GetImg_maxWH(), 0.0);

	Cylinder curCylinder;
	curCylinder.center = center;
	curCylinder.radius = 1.5 / image.GetImg_maxWH();
	curCylinder.height = 2 * H;

	BiSectionMethod(x_index, y_index, curCylinder, image, sky, sun);

	return curCylinder;
}

void Cloud::BiSectionMethod(int px, int py, Cylinder & curCylinder, Image image, Sky sky, Sun sun)
{
	float x = px*1.0 / image.GetImg_maxWH();
	float y = (image.GetImg_height() - py)*1.0 / image.GetImg_maxWH();

	float left_H = 0;
	float right_H = 1;
	float left_value = ComputeSingleScattering(left_H, px, py , image ,sky, sun);
	float right_value = ComputeSingleScattering(right_H, px, py ,image ,sky, sun);

	int loop = 0;
	while (right_value - left_value > 0.005&& loop < 15)
	{
		float curH = (left_H + right_H) / 2;
		float singlevalue = ComputeSingleScattering(curH, px, py, image, sky, sun);
		if (singlevalue > image.GetImg_mat()[py*image.GetImg_width() + px])
		{
			right_value = singlevalue;
			right_H = curH;

		}
		if (singlevalue < image.GetImg_mat()[py*image.GetImg_width() + px])
		{
			left_value = singlevalue;
			left_H = curH;
		}

		loop++;
	}
	float ret_H = (left_H + right_H);
	curCylinder.height = ret_H;
}

void Cloud::PropagationCylinders(Image image, Sky sky, Sun sun)
{
	int count = 0;
	for (int i = 0; i<path.size(); i++)
	{
		int x_index = path[i].x;
		int y_index = path[i].y;
		Cylinder curCylinder = CreateCylinder(x_index, y_index, image, sky, sun);
		curCloudVolume.push_back(curCylinder);
		count++;
		if (count % 100 == 0) {
			//cout << "Remain: " << path.size() - count << "  (" << x_index << "," << y_index << ")   " << curCylinder.height / 2 << endl;
			curCylinder.height / 2;
		}
			
		cloudIn.Update(curCylinder);

	}
}

float Cloud::ComputeSingleScattering(float H, int px, int py, Image image, Sky sky , Sun sun)
{
	float integralvalue = 0.0;//积分值
	float dH = 2 * H / (INT_RES - 1);//INT_RES = 50;
	float x = px*1.0 / image.GetImg_maxWH();
	float y = (image.GetImg_height() - py)*1.0 / image.GetImg_maxWH();

	float totalTrans = 1.0;
	float  curTrans = expf(-CONSTANT_ATTEN*dH);

	Cylinder extraLocalVolume;
	extraLocalVolume.center = Vector3(x, y, 0);
	extraLocalVolume.height = 2 * H;
	extraLocalVolume.radius = 1.5 / image.GetImg_maxWH();

	for (int i = 0; i<INT_RES; i++)
	{
		float zi = -H + dH*i;

		Vector3 Xi(x, y, zi);
		float pathlen = cloudIn.PathLen(Xi, sun.GetSunVecUV(), extraLocalVolume);//在cloud里面计算出一个浮点型的lignt，
		integralvalue += pathlen*totalTrans*(1 - curTrans)*PhaseFunction(Vector3(0, 0, -1), Normalize(sun.GetSunVecUV()));
		totalTrans *= curTrans;

	}


	integralvalue += sky.GetImg_grey_sky()[py*image.GetImg_width() + px] * totalTrans + AMBIENT*(1 - expf(-2 * H*CONSTANT_ATTEN));
	//cout<<"scattervalue"<<integralvalue<<endl;
	return  integralvalue;
}

float Cloud::ComputeNormalHeightNeightborAvg(int px, int py, Image image, Pixel pixel, Sky sky)
{
	int count = 0;
	float avg = 0.0;
	int x_index = px;
	int y_index = py;
	float radius = 1;
	while (count == 0 && radius<20)
	{
		for (int i = -radius; i <= radius; i++)
			for (int j = -radius; j <= radius; j++)
			{
				if (x_index + j >= 0 && y_index + i >= 0 && x_index + j<image.GetImg_width()&&y_index + i<image.GetImg_height())
				{
					if (pixel.GetPixelTypeList()[(y_index + i)*image.GetImg_width() + x_index + j] == 1 && sky.GetImg_grey_sky()[(y_index + i)*image.GetImg_width() + x_index + j] - image.GetImg_mat()[(y_index + i)*image.GetImg_width() + x_index + j]<0)
					{
						avg += heightField[(y_index + i)*image.GetImg_width() + x_index + j];
						count++;

					}
				}
			}

		if (count>0)
		{

			avg /= count;

		}

		radius += 2;

	}


	return  avg;
}

float Cloud::PhaseFunction(Vector3 v1, Vector3 v2)
{
	float cosTheta = Angle(v1, v2);
	//float g=0.85;  
	//	return 1.0/ (4.0 * M_PI) * (1.0 - g*g) /pow(1.0 + g*g - 2.0 * g * cosTheta, 1.5);

	return 0.75*(1 + cosTheta*cosTheta) / (4.0 * M_PI);
}

void Cloud::CreateHeightFieldHalf(Image image,Pixel pixel,Sky sky)
{
	heightField = new float[image.GetImg_height()*image.GetImg_width()];
	if (heightField == NULL)
	{
		cout << "allocate height field failed\n";
		exit(1);

	}

	cloudIn.CreatHeightField();

	for (int i = 0; i<image.GetImg_height(); i++)
	{
		float delta_H = 0;
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1)
			{
				float x = 1.0*j / image.GetImg_maxWH();
				float y = 1.0*(image.GetImg_height() - i) / image.GetImg_maxWH();

				heightField[i*image.GetImg_width() + j] = cloudIn.Interpolat(x, y);//+delta_H;

			}
			else
			{
				heightField[i*image.GetImg_width() + j] = 0.0;
			}

		}

	}


	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1)
			{
				if (sky.GetImg_grey_sky()[i*image.GetImg_width() + j] - image.GetImg_mat()[i*image.GetImg_width() + j]>0)  //set the height of  a singular pixel  to  the average of heights of neighboring pixels.
				{
					heightField[i*image.GetImg_width() + j] = ComputeNormalHeightNeightborAvg(j, i, image, pixel, sky);

				}

			}
		}
	/*
	ofstream out("../output/img_height.txt");
	//out.open("clinder_height.txt");
	for (int i = 0; i<image.GetImg_height(); i++)
	{
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			out << heightField[i*image.GetImg_width() + j] << " ";
			if (i % 10 == 0 && j % 10 == 0) {
				printf("heightField[%d*img_width+%d]=%f\n", i, j, heightField[i*image.GetImg_width() + j]);
			}

		}

		out << endl;

	}
	*/
	//cv::Mat img(image.GetImg_height(), image.GetImg_width(),CV_8UC1);
	//for (int i = 0; i < image.GetImg_height(); ++i) {
	//	for (int j = 0; j < image.GetImg_width(); ++j) {
	//		img.at<uchar>(i,j) = static_cast<uchar>(heightField[i * image.GetImg_width() + j]*255);
	//	}
	//}
	//cv::namedWindow("Display window", cv::WINDOW_AUTOSIZE);
	//cv::imshow("Display window", img);
	//cv::waitKey(0);
	//out.close();
}

//--------------------------------------
void Cloud::CreateHeightFieldOtherHalf(Image image, Pixel pixel, Perlin* perlin)
{
	int x_min = 9999;
	int x_max = -9999;
	int y_min = 9999;
	int y_max = -9999;
	for (int i = 0; i<image.GetImg_height(); i++)
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == -1)
			{
				if (x_max<j)
					x_max = j;
				if (x_min>j)
					x_min = j;
				if (y_max<i)
					y_max = i;
				if (y_min>i)
					y_min = i;
			}
		}

	int x_middle = (x_min + x_max) / 2;
	otherHeightField = new float[image.GetImg_height()*image.GetImg_width()];

	if (otherHeightField == NULL)
	{
		cout << "allocate other height field failed\n";
		exit(1);

	}


	for (int i = 0; i<image.GetImg_height(); i++)
	{
		int left = 999999;
		int right = -1;

		//float scale=0.5*exp(1.0*i/(img_height-1)-1);
		for (int j = 0; j<image.GetImg_width(); j++)
		{
			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1)
			{
				if (j>right)
					right = j;
				if (j<left)
					left = j;

			}

		}

		float scale = 0.5* /*exp(1.5*(right-left)/(x_max-x_min)-1)*/expf(1.0*(i - y_min) / (y_max - y_min) - 1);
		for (int j = 0; j<image.GetImg_width(); j++)
		{

			if (pixel.GetPixelTypeList()[i*image.GetImg_width() + j] == 1)
			{
				float disturb = expf((perlin->Get(i*1.0 / image.GetImg_height(), j*1.0 / image.GetImg_width()) + 1.0) / 2) / expf(1);
				float myfun;
				if (j<x_middle)
					myfun = (j - left)*(2 * x_middle - left - j) / pow(x_middle - left, 2.0);
				else
					myfun = (j - right)*(2 * x_middle - right - j) / pow(right - x_middle, 2.0);
				otherHeightField[i*image.GetImg_width() + j] = heightField[i*image.GetImg_width() + j] - exp(1.2*heightField[i*image.GetImg_width() + j])*disturb*scale*myfun;

			}
			else
			{
				otherHeightField[i*image.GetImg_width() + j] = 0.0;
			}

		}
	}
}
//--------------------------------------


void Cloud::SetHeightField(int i, float value)
{
	heightField[i] = value;
}

//--------------
void Cloud::RemoveOneLoopPixelBoudary(Image image, Pixel pixel, float* heightField){
	//int*  tempPixelList = new int[img_width*img_height];
	int*  tempPixelList = new int[image.GetImg_width()*image.GetImg_height()];

	for (int i = 0; i<image.GetImg_height(); i++)
	for (int j = 0; j<image.GetImg_width(); j++)
	{
		tempPixelList[i*image.GetImg_width() + j] = pixel.pixelTypeList[i*image.GetImg_width() + j];
	}

	for (int i = 0; i<image.GetImg_height(); i++)
	for (int j = 0; j<image.GetImg_width(); j++)
	{
		if (tempPixelList[i*image.GetImg_width() + j] == 1)
		{

			for (int ii = -1; ii<2; ii++)
			for (int jj = -1; jj<2; jj++)
			{
				if ((i + ii)<image.GetImg_height() && (i + ii) >= 0 && (j + jj)<image.GetImg_width() && (j + jj) >= 0 && tempPixelList[(i + ii)*image.GetImg_width() + j + jj] == -1)
				{
					pixel.pixelTypeList[i*image.GetImg_width() + j] = -1;
					heightField[i*image.GetImg_width() + j] = 0.0;
					break;
				}
			}


		}
	}
	delete[] tempPixelList;
}
//--------------

//--------------
float Cloud::InterPolateHeightField(Image image, Pixel pixel, Sky sky, float x, float y, float* heightField){
	if (x<0 || x>1)
		return 0;
	if (y<0 || y>1)
		return 0;

	float interval_x = 1.0 /  image.img_maxWH;
	float interval_y = 1.0 / image.img_maxWH;
	int x_index = int((x - 0) / interval_x);
	int y_index = image.img_height - int((y - 0) / interval_y);
	//int x_index = int(round(x));
	//int y_index = image.img_height - int(round(y));

	float height = 0.0;
	float weightSum = 0.0;
	float radius = 1;

	float isFind = false;
	while (!isFind)
	{
		int  count = 0;
		for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			if (x_index + j >= 0 && y_index + i >= 0 && x_index + j<image.img_width&&y_index + i<image.img_height&&pixel.pixelTypeList[(y_index + i)*image.img_width + x_index + j] == 1)
			{
				if (image.img_mat[y_index*image.img_width + x_index]>sky.img_grey_sky[y_index*image.img_width + x_index])
				{
					float  disSqr = i*i + j*j;
					float weight = exp(-disSqr / (radius*radius));
					weightSum += weight;
					height += weight*heightField[(y_index + i)*image.img_width + x_index + j];
					count++;
				}

			}
		}

		if (count>0)
		{
			height /= weightSum;
			isFind = true;
		}

		radius += 1;

		if (radius > 50)
		{
			height = 0;
			break;

		}

	}
	return height;
}

void Cloud::NormalizeCloudMesh(Mesh mesh){
	float center_x = 0, center_y = 0, center_z = 0;
	float min_x = 9999, min_y = 9999, min_z = 9999;
	float max_x = -9999, max_y = -9999, max_z = -9999;

	for (int i = 0; i<mesh.Cloud_vertexnumber; i++)
	{
		float x = mesh.Cloud_vertexList[3 * i + 0];
		float y = mesh.Cloud_vertexList[3 * i + 1];
		float z = mesh.Cloud_vertexList[3 * i + 2];
		center_x += x;
		center_y += y;
		center_z += z;
		if (min_x>x)
			min_x = x;
		if (max_x<x)
			max_x = x;
		if (min_y>y)
			min_y = y;
		if (max_y<y)
			max_y = y;
		if (min_z>z)
			min_z = z;
		if (max_z<z)
			max_z = z;
	}
	center_x /= mesh.Cloud_vertexnumber;
	center_y /= mesh.Cloud_vertexnumber;
	center_z /= mesh.Cloud_vertexnumber;

	float half_x_extent = (max_x - min_x) / 2;
	float half_y_extent = (max_y - min_y) / 2;
	float half_z_extent = (max_z - min_z) / 2;

	float half_max = half_x_extent;
	if (half_max<half_y_extent)
		half_max = half_y_extent;
	if (half_max<half_z_extent)
		half_max = half_z_extent;

	float  scale = 1.0;
	for (int i = 0; i<mesh.Cloud_vertexnumber; i++)
	{
		mesh.Cloud_vertexList[3 * i + 0] = (mesh.Cloud_vertexList[3 * i + 0] - center_x) / half_max*scale;
		mesh.Cloud_vertexList[3 * i + 1] = (mesh.Cloud_vertexList[3 * i + 1] - center_y) / half_max*scale;
		mesh.Cloud_vertexList[3 * i + 2] = (mesh.Cloud_vertexList[3 * i + 2] - center_z) / half_max*scale;
	}

}

void Cloud::CreateCloudMesh(Image image, Pixel pixel, Sky sky, Mesh &mesh, float* heightField){

	mesh.Cloud_facenumber = mesh.face_number;
	mesh.Cloud_vertexnumber = mesh.ver_number;

	mesh.Cloud_vertexList = new float[3 * mesh.Cloud_vertexnumber];
	if (mesh.Cloud_vertexList == NULL)
	{
		cout << "allocate Cloud vetex failed\n";
		exit(1);

	}

	mesh.Cloud_facelist = new int[3 * mesh.Cloud_facenumber];
	if (mesh.Cloud_facelist == NULL)
	{
		cout << "allocate Cloud face failed\n";
		exit(1);

	}

	//Vertex 
	//for top
	for (int i = 0; i<mesh.ver_number; i++)
	{
		mesh.Cloud_vertexList[3 * i + 0] = mesh.vertexList[3 * i + 0] / image.img_maxWH;
		mesh.Cloud_vertexList[3 * i + 1] = mesh.vertexList[3 * i + 1] / image.img_maxWH;
		mesh.Cloud_vertexList[3 * i + 2] = InterPolateHeightField(image, pixel, sky, mesh.Cloud_vertexList[3 * i + 0], mesh.Cloud_vertexList[3 * i + 1], heightField);
		//cout << mesh.Cloud_vertexList[3 * i + 0] << "  " << mesh.Cloud_vertexList[3 * i + 1] << " " << mesh.Cloud_vertexList[3 * i + 2] << endl;
	}


	//Face

	//for top
	for (int i = 0; i<mesh.face_number; i++)
	{
		mesh.Cloud_facelist[3 * i + 0] = mesh.faceList[4 * i + 1];
		mesh.Cloud_facelist[3 * i + 1] = mesh.faceList[4 * i + 2];
		mesh.Cloud_facelist[3 * i + 2] = mesh.faceList[4 * i + 3];
	}


	//NormalizeCloudMesh(mesh);
}
//--------------

//--------------
void Cloud::ExportCloudMesh(Mesh& mesh, char* filename){
	ofstream out(filename);
	out << "OFF" << endl;
	out << mesh.Cloud_vertexnumber << " " << mesh.Cloud_facenumber << " " << 0 << endl;
	for (int i = 0; i<mesh.Cloud_vertexnumber; i++)
	{
		float x = mesh.Cloud_vertexList[3 * i + 0];
		float y = mesh.Cloud_vertexList[3 * i + 1];
		float z = mesh.Cloud_vertexList[3 * i + 2];

		out << x << " " << y << " " << z << endl;

	}

	for (int i = 0; i<mesh.Cloud_facenumber; i++)
	{

		out << 3 << " " << mesh.Cloud_facelist[3 * i + 0] << " " << mesh.Cloud_facelist[3 * i + 1] << " " << mesh.Cloud_facelist[3 * i + 2] << endl;

	}
}
//--------------
