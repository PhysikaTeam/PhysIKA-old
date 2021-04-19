#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<iterator>
#include"ConvertParticlesToVoxel.h"
#include"WriteVTI.hpp"
#include"vector_types.h"
using namespace std;


//extern "C" void Dis(const float* const vertices, float* const dis, const int& nTri, int step);
//采样粒子到规矩网格
extern "C" void _transformParIntoGrid(float *gridDensity, float *density, float *mass, float *pos, 
	float xResolution, float yResolution, float zResolution, 
	float domain_max_x, float domain_max_y, float domain_max_z, 
	float domain_min_x, float domain_min_y, float domain_min_z, 
	float step, float h, unsigned int numParticle);


//void Dis(const float* const vertices, float* const dis, const int& nTri, int step)
//{
//	return;
//}
//
//void _transformParIntoGrid(float *gridDensity, float *density, float *mass, float *pos,
//	float xResolution, float yResolution, float zResolution,
//	float domain_max_x, float domain_max_y, float domain_max_z,
//	float domain_min_x, float domain_min_y, float domain_min_z,
//	float step, float h, unsigned int numParticle)
//{
//	return;
//}


void transformParticleIntoGrid(const string& inputName, const string& outputName, int RESOLUTION = 64, float particleRadius = 0.02)
{
	char m_inputName[256];
	stringstream sstr;
	sstr << inputName;
	sstr >> m_inputName;
	cout << inputName << endl;
	//---------------read particle file
	FILE *in;
	if ((in = fopen(m_inputName, "rb")) == NULL)
	{
		std::cerr << "Can not open file for read : " << m_inputName << std::endl;
		return;
	}

	unsigned int particleNumber; //粒子数

	float max_x, max_y, max_z;
	float min_x, min_y, min_z;
	max_x = -9999.0;
	max_y = -9999.0;
	max_z = -9999.0;
	min_x = 9999.0;
	min_y = 9999.0;
	min_z = 9999.0;

	std::vector<float3> particlePos; //粒子位置
	std::vector<float> particleDensity; //粒子密度
	std::vector<float> particleMass; //粒子质量
	fread(&particleNumber, sizeof(int), 1, in);

	//初始化容器大小
	particlePos.resize(particleNumber);
	particleDensity.resize(particleNumber);
	particleMass.resize(particleNumber);

	//read position 
	float temX, temY, temZ;

	for (int i = 0; i < particleNumber; i++)
	{
		fread(&temX, sizeof(float), 1, in);
		fread(&temY, sizeof(float), 1, in);
		fread(&temZ, sizeof(float), 1, in);

		particlePos[i].x = temX;
		particlePos[i].y = temY;
		particlePos[i].z = temZ;

		//用于云生成时的坐标读取，需将坐标位置从0-64映射到-1-1
		/*particlePos[i].x = (temX-32)/32;
		particlePos[i].y = (temY-32)/32;
		particlePos[i].z = (temZ-32)/32;*/

		/*if (particlePos[i].x < -1 || particlePos[i].y < -1 || particlePos[i].z < -1 || particlePos[i].x>1 || particlePos[i].y>1 || particlePos[i].z>1)
		{
			std::cout << particlePos[i].x << "  " << particlePos[i].y << "  " << particlePos[i].z << std::endl;
		}*/

		temX = particlePos[i].x;
		temY = particlePos[i].y;
		temZ = particlePos[i].z;

		//compute the range(范围）
		if (temX > max_x)
			max_x = temX;
		if (temX < min_x)
			min_x = temX;

		if (temY > max_y)
			max_y = temY;
		if (temY < min_y)
			min_y = temY;

		if (temZ > max_z)
			max_z = temZ;
		if (temZ < min_z)
			min_z = temZ;
	}
	/*std::cout << max_x << "  " << min_x << std::endl;
	std::cout << max_y << "  " << min_y << std::endl;
	std::cout << max_z << "  " << min_z << std::endl;*/

	//read density
	float temDensity;
	for (int i = 0; i < particleNumber; i++)
	{
		fread(&temDensity, sizeof(float), 1, in);
		particleDensity[i] = temDensity;
		//particleDensity[i] = 0.25;
		//std::cout << temDensity << std::endl;
	}

	//read mass
	for (int i = 0; i < particleNumber; i++)
	{
		particleMass[i] = 0.0512;
	}


	//-----------------------------------------------------------------------------------------------------
	//-------------生成规则网格-即网格分辨率固定为：n*n*n-------例如:128*128*128----------start---------------
	//----------------------------------------2019-5-26----------------------------------------------------
	int xResolution, yResolution, zResolution;
	float domain_max_x, domain_max_y, domain_max_z;
	float domain_min_x, domain_min_y, domain_min_z;

	//定义网格分辨率
	xResolution = RESOLUTION;
	yResolution = RESOLUTION;
	zResolution = RESOLUTION;


	//定义网格的x/y/z的长度
	//float domain_length = (RESOLUTION * gridSize) / 2;
	float domain_length = (RESOLUTION * 2 * particleRadius) / 2;
	domain_max_x = domain_length;
	domain_max_y = domain_length;
	domain_max_z = domain_length;

	domain_min_x = -1.0 * domain_length;
	domain_min_y = -1.0 * domain_length;
	domain_min_z = -1.0 * domain_length;


	//设定网格域的范围,网格间距为2*粒子半径=0.04，有128*0.04/2=2.56
	float gridSize;
	//gridSize = 2 * particleRadius;
	gridSize = (domain_max_x - domain_min_x) / xResolution;


	//std::cout << "new grid: " << xResolution << " " << domain_max_x << " " << domain_min_y << std::endl;
	//-----------------------------------------------------------------------------------------------------
	//-------------生成规则网格-----------------------n*n*n-------例如:128*128*128----------end--------------
	//----------------------------------------2019-5-26----------------------------------------------------


	/////---------------compute density---------
	float h = 4 * gridSize;
	int fSize = xResolution * yResolution * zResolution;
	float *gridDensity = new float[fSize];

	_transformParIntoGrid(gridDensity, particleDensity.data(), particleMass.data(), (float*)particlePos.data(), xResolution, yResolution, zResolution, domain_max_x, domain_max_y, domain_max_z, domain_min_x, domain_min_y, domain_min_z, gridSize, h, particleNumber);

	vector<float> tmp(gridDensity, gridDensity+fSize);
	WriteVTI(xResolution-1, yResolution-1, zResolution-1, tmp, outputName);

	std::cout << "OVER" << endl;
}



void transformParticleIntoGrid(std::vector<float3>& particlePos,
							std::vector<float>& particleDensity,
							std::string& outputName,
							int RESOLUTION = 64, 
							float particleRadius = 0.02)
{
	float max_x, max_y, max_z;
	float min_x, min_y, min_z;
	max_x = -9999.0;
	max_y = -9999.0;
	max_z = -9999.0;
	min_x = 9999.0;
	min_y = 9999.0;
	min_z = 9999.0;

	int particleNumber = particlePos.size();

	std::vector<float> particleMass(particleNumber, 0.0512); //粒子质量

	//read position 
	float temX, temY, temZ;

	for (int i = 0; i < particleNumber; i++)
	{
		temX = particlePos[i].x;
		temY = particlePos[i].y;
		temZ = particlePos[i].z;

		//compute the range(范围）
		if (temX > max_x)
			max_x = temX;
		if (temX < min_x)
			min_x = temX;

		if (temY > max_y)
			max_y = temY;
		if (temY < min_y)
			min_y = temY;

		if (temZ > max_z)
			max_z = temZ;
		if (temZ < min_z)
			min_z = temZ;
	}


	//-----------------------------------------------------------------------------------------------------
	//-------------生成规则网格-即网格分辨率固定为：n*n*n-------例如:128*128*128----------start---------------
	//----------------------------------------2019-5-26----------------------------------------------------
	int xResolution, yResolution, zResolution;
	float domain_max_x, domain_max_y, domain_max_z;
	float domain_min_x, domain_min_y, domain_min_z;

	//定义网格分辨率
	xResolution = RESOLUTION;
	yResolution = RESOLUTION;
	zResolution = RESOLUTION;


	//定义网格的x/y/z的长度
	//float domain_length = (RESOLUTION * gridSize) / 2;
	float domain_length = (RESOLUTION * 2 * particleRadius) / 2;
	domain_max_x = domain_length;
	domain_max_y = domain_length;
	domain_max_z = domain_length;

	domain_min_x = -1.0 * domain_length;
	domain_min_y = -1.0 * domain_length;
	domain_min_z = -1.0 * domain_length;


	//设定网格域的范围,网格间距为2*粒子半径=0.04，有128*0.04/2=2.56
	float gridSize;
	//gridSize = 2 * particleRadius;
	gridSize = (domain_max_x - domain_min_x) / xResolution;


	/////---------------compute density---------
	float h = 4 * gridSize;
	int fSize = xResolution * yResolution * zResolution;
	float *gridDensity = new float[fSize];

	_transformParIntoGrid(gridDensity, particleDensity.data(), particleMass.data(), (float*)particlePos.data(), xResolution, yResolution, zResolution, domain_max_x, domain_max_y, domain_max_z, domain_min_x, domain_min_y, domain_min_z, gridSize, h, particleNumber);

	vector<float> tmp(gridDensity, gridDensity + fSize);
	WriteVTI(xResolution - 1, yResolution - 1, zResolution - 1, tmp, outputName);

	std::cout << "OVER" << endl;
}




//int main()
//{
//	std::string inputName("D:\\Code\\PositionBasedDynamics-master\\CloudResults\\tran-scale-dragon-new_tran-scale-bunny\\000000.bin");
//	//inputName = "D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-dragon.dat";
//
//	std::string outputName("D:\\Code\\PositionBasedDynamics-master\\cloud4learning\\tran-scale-dragon.vti");
//
//	transformParticleIntoGrid(inputName, outputName);
//	return 0;
//}