#include "CloudInterior.h"

CloudInterior::CloudInterior(void)
{
	interval_x = (CloudBox.x_max - CloudBox.x_min) / (CloudINTERIOR_RES - 1);
	interval_y = (CloudBox.y_max - CloudBox.y_min) / (CloudINTERIOR_RES - 1);
	interval_z = (CloudBox.z_max - CloudBox.z_min) / (CloudINTERIOR_RES - 1);

	isInCloud = new int[CloudINTERIOR_RES*CloudINTERIOR_RES*CloudINTERIOR_RES];

	if (isInCloud == NULL)
	{
		cout << "isInCloud memory allocated failure!" << endl;
		exit(0);
	}

	for (int i = 0; i<CloudINTERIOR_RES*CloudINTERIOR_RES*CloudINTERIOR_RES; i++)
	{
		isInCloud[i] = 0;
	}


	heightField = new float[CloudINTERIOR_RES*CloudINTERIOR_RES];
	if (heightField == NULL)
	{
		cout << "height field for class  cloudin  memory allocated failure!" << endl;
		exit(0);
	}

	for (int i = 0; i<CloudINTERIOR_RES*CloudINTERIOR_RES; i++)
	{
		heightField[i] = 0;
	}
}
CloudInterior::~CloudInterior(void)
{
	if (isInCloud != NULL)
		delete[] isInCloud;


	if (heightField != NULL)
		delete[]heightField;


}


bool CloudInterior::IsCloudCube(int x, int y, int z)
{

	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
			for (int k = -1; k <= 1; k++)
			{
				if (z + i >= 0 && z + i<CloudINTERIOR_RES&&y + j >= 0 && y + j<CloudINTERIOR_RES&&x + k >= 0 && x + k<CloudINTERIOR_RES)
				{
					if (isInCloud[(z + i)*CloudINTERIOR_RES*CloudINTERIOR_RES + (y + j)*CloudINTERIOR_RES + (x + k)] == 1)
						return  true;
				}
			}

	return false;

}


void CloudInterior::CreatHeightField()
{

	for (int i = 0; i<CloudINTERIOR_RES; i++)
		for (int j = 0; j<CloudINTERIOR_RES; j++)
		{

			heightField[i*CloudINTERIOR_RES + j] = 0;

		}
	for (int i = 0; i<CloudINTERIOR_RES; i++)
		for (int j = 0; j<CloudINTERIOR_RES; j++)
		{
			int count = 0;
			for (int k = 0; k<CloudINTERIOR_RES; k++)
			{
				if (isInCloud[k*CloudINTERIOR_RES*CloudINTERIOR_RES + i*CloudINTERIOR_RES + j] == 1)
				{
					count++;
				}

			}

			heightField[i*CloudINTERIOR_RES + j] = count*interval_z / 2;

		}


	////scale  the height field  to [0,0.5]
	//float MaxHeight=-9999;
	//float MinHeight = 9999;
	//for(int i=0;i<CloudINTERIOR_RES;i++)
	//	for(int j=0;j<CloudINTERIOR_RES;j++)
	//	{
	//		if(MaxHeight<heightField[i*CloudINTERIOR_RES+j])
	//			MaxHeight=heightField[i*CloudINTERIOR_RES+j];
	//		if(MinHeight>heightField[i*CloudINTERIOR_RES+j])
	//		    MinHeight=heightField[i*CloudINTERIOR_RES+j];

	//	}
	//	for(int i=0;i<CloudINTERIOR_RES;i++)
	//		for(int j=0;j<CloudINTERIOR_RES;j++)
	//		{
	//			heightField[i*CloudINTERIOR_RES+j]=0.5*(heightField[i*CloudINTERIOR_RES+j]-MinHeight)/(MaxHeight-MinHeight);
	//		}


}

float CloudInterior::Interpolat(float x, float y)
{
	if (x<CloudBox.x_min || x>CloudBox.x_max)
		return 0;
	if (y<CloudBox.y_min || y>CloudBox.y_max)
		return 0;

	int x_index = int((x - CloudBox.x_min) / interval_x);
	int y_index = int((y - CloudBox.y_min) / interval_y);

	int count = 0;
	float height = 0.0;
	for (int i = -1; i <= 1; i++)
		for (int j = -1; j <= 1; j++)
		{
			if (x_index + j >= 0 && y_index + i >= 0 && x_index + j<CloudINTERIOR_RES&&y_index + i<CloudINTERIOR_RES)
			{
				height += heightField[(y_index + i)*CloudINTERIOR_RES + x_index + j];
				count++;
			}
		}

	if (count>0)
	{
		height /= count;
	}

	return height;

}

void CloudInterior::Update(const Cylinder& curCylinder)
{
	for (float cur_x = curCylinder.center.x - curCylinder.radius; cur_x<curCylinder.center.x + curCylinder.radius; cur_x += interval_x)
	{
		float y_start = curCylinder.center.y - sqrt(curCylinder.radius*curCylinder.radius - pow(cur_x - curCylinder.center.x, 2));
		float y_end = curCylinder.center.y + sqrt(curCylinder.radius*curCylinder.radius - pow(cur_x - curCylinder.center.x, 2));
		for (float cur_y = y_start; cur_y<y_end; cur_y += interval_y)
		{
			float z_start = curCylinder.center.z - curCylinder.height / 2/*-sqrt(curCylinder.radius*curCylinder.radius-pow(cur_x-curCylinder.center.x,2)-pow(cur_y-curCylinder.center.y,2))*/;
			float z_end = curCylinder.center.z + curCylinder.height / 2/*+sqrt(curCylinder.radius*curCylinder.radius-pow(cur_x-curCylinder.center.x,2)-pow(cur_y-curCylinder.center.y,2))*/;

			for (float cur_z = z_start; cur_z<z_end; cur_z += interval_z)
			{
				int x_index = int((cur_x - CloudBox.x_min) / interval_x);
				int y_index = int((cur_y - CloudBox.y_min) / interval_y);
				int z_index = int((cur_z - CloudBox.z_min) / interval_z);

				if (x_index >= 0 && y_index >= 0 && z_index >= 0 && x_index<CloudINTERIOR_RES&&y_index<CloudINTERIOR_RES&&z_index<CloudINTERIOR_RES)
					isInCloud[z_index*CloudINTERIOR_RES*CloudINTERIOR_RES + y_index*CloudINTERIOR_RES + x_index] = 1;

			}
		}


	}

}


bool CloudInterior::FindSegment(float intesection[2], float* P, float* direction)
{
	float x_min = CloudBox.x_min;
	float x_max = CloudBox.x_max;
	float y_min = CloudBox.y_min;
	float y_max = CloudBox.y_max;
	float z_min = CloudBox.z_min;
	float z_max = CloudBox.z_max;
	int nCross = 0;
	//front 
	if (fabs(direction[0])>F_ZERO)//F_zero代表的是一个浮点0
	{
		float t = (x_max - P[0]) / direction[0];
		float y = P[1] + t*direction[1];
		float z = P[2] + t*direction[2];
		if (y >= y_min&&y <= y_max&&z <= z_max&&z >= z_min)
		{
			intesection[nCross] = t;
			nCross++;
		}
		if (nCross >= 2)
			return true;

	}
	//back 
	if (fabs(direction[0])>F_ZERO)
	{
		float t = (x_min - P[0]) / direction[0];
		float y = P[1] + t*direction[1];
		float z = P[2] + t*direction[2];
		if (y >= y_min&&y <= y_max&&z <= z_max&&z >= z_min)
		{
			intesection[nCross] = t;
			nCross++;
		}
		if (nCross >= 2)
			return true;

	}
	//left 
	if (fabs(direction[1])>F_ZERO)
	{
		float t = (y_min - P[1]) / direction[1];
		float x = P[0] + t*direction[0];
		float z = P[2] + t*direction[2];
		if (x >= x_min&&x <= x_max&&z <= z_max&&z >= z_min)
		{
			intesection[nCross] = t;
			nCross++;
		}
		if (nCross >= 2)
			return true;

	}

	//right
	if (fabs(direction[1])>F_ZERO)
	{
		float t = (y_max - P[1]) / direction[1];
		float x = P[0] + t*direction[0];
		float z = P[2] + t*direction[2];
		if (x >= x_min&&x <= x_max&&z <= z_max&&z >= z_min)
		{
			intesection[nCross] = t;
			nCross++;
		}
		if (nCross >= 2)
			return true;

	}
	//top
	if (fabs(direction[2])>F_ZERO)
	{
		float t = (z_max - P[2]) / direction[2];
		float x = P[0] + t*direction[0];
		float y = P[1] + t*direction[1];
		if (y >= y_min&&y <= y_max&&x <= x_max&&x >= x_min)
		{
			intesection[nCross] = t;
			nCross++;
		}
		if (nCross >= 2)
			return true;

	}
	//bottom
	if (fabs(direction[2])>F_ZERO)
	{
		float t = (z_min - P[2]) / direction[2];
		float x = P[0] + t*direction[0];
		float y = P[1] + t*direction[1];
		if (y >= y_min&&y <= y_max&&x <= x_max&&x >= x_min)
		{
			intesection[nCross] = t;
			nCross++;
		}
		if (nCross >= 2)
			return true;

	}

	if (nCross<2)
		return  false;

}
float CloudInterior::PathLen(Vector3 P0, Vector3 direction, const  Cylinder&  extraLocalVolume)
{
	//find segment which is in [x_min,x_max;y_min,y_max;z_min,z_max], max {left}<=t<=min{left}
	float intersection[2];
	bool  isFind = FindSegment(intersection, !P0, !direction);//操作是将三阶点返回一个浮点型数组

	if (!isFind)
	{
		return 0.0;
	}

	float t0 = min(intersection[0], intersection[1]);
	float t1 = max(intersection[0], intersection[1]);

	//note here 
	t0 = 0.0;

	int LineSampleRes = INT_RES;
	float line_interval = (t1 - t0) / (LineSampleRes - 1);
	float sampleInterval = Magnitude(direction)*(t1 - t0) / (LineSampleRes - 1);

	float Trans = expf(-sampleInterval*CONSTANT_ATTEN);
	float  light = SUN_INTENSITY;
	float scale = 1.5*SOLID_ANGLE / (4 * M_PI);

	int D = 0;//这个D变量命名就很有问题看不太懂
	float q = ((1 - Trans)*scale + Trans);
	for (int i = LineSampleRes - 2; i>0; i--)
	{
		Vector3 samplePoint = P0 + direction*(t0 + (i + 1)*line_interval);
		int x_index = int((samplePoint.x - CloudBox.x_min) / interval_x);
		int y_index = int((samplePoint.y - CloudBox.y_min) / interval_y);
		int z_index = int((samplePoint.z - CloudBox.z_min) / interval_z);

		if (IsCloudCube(x_index, y_index, z_index)/*||isInLocalVolume(P0,extraLocalVolume)*/)
		{
			/*	light*= ((1-Trans)*scale+Trans);*/
			D++;
			//if(! isInLocalVolume(P0,extraLocalVolume))
			//             cout<<" not in the extra local volume "<<endl;

		}

	}

	light = SUN_INTENSITY*expf((scale - 1)*D*sampleInterval*CONSTANT_ATTEN);
	//cout<<D<<endl;

	//light=SUN_INTENSITY*powf(q,D);
	return light;

}

bool CloudInterior::isInLocalVolume(Vector3 p0, const Cylinder& curCylinder)
{
	float x = p0.x;
	float y = p0.y;
	float z = p0.z;

	if (z<curCylinder.center.z - curCylinder.height / 2)
		return false;

	if (z>curCylinder.center.z + curCylinder.height / 2)
		return false;

	float dis2center = pow(x - curCylinder.center.x, 2) + pow(y - curCylinder.center.y, 2);
	if (dis2center>curCylinder.radius*curCylinder.radius)
		return false;

	return  true;


}
