#pragma once
#include "SatDataCumulus.h"
#include "global.h"
#include "math.h"
#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/lloyd_optimize_mesh_2.h>
using namespace std;

SatDataCloud::SatDataCloud(
	int TOPLEFT_X,
	int TOPLEFT_Y,
	int REGION_WIDTH,
	int REGION_HEIGHT,
	double SCALE_EOFF,
	string LANDSAT_SCENE_ID,
	long long FILE_DATE,
	int YEAR,
	int MONTH,
	int DAY,
	int HOUR,
	int MINUTE,
	int SECONDS,
	int LAND_AVG_TEMPERATURE,
	string SPACECRAFT_ID,
	string SENSOR_ID,
	int WRS_PATH,
	int WRS_ROW,
	string FILE_NAME_BAND_1,
	string FILE_NAME_BAND_2,
	string FILE_NAME_BAND_3,
	string FILE_NAME_BAND_4,
	string FILE_NAME_BAND_5,
	string FILE_NAME_BAND_6,
	string FILE_NAME_BAND_7,
	string FILE_NAME_BAND_8,
	string FILE_NAME_BAND_9,
	string FILE_NAME_BAND_10,
	string FILE_NAME_BAND_11,
	string FILE_NAME_BAND_QUALITY,
	string METADATA_FILE_NAME
)
{
	this->TOPLEFT_X = TOPLEFT_X;
	this->TOPLEFT_Y = TOPLEFT_Y;
	this->REGION_WIDTH = REGION_WIDTH;
	this->REGION_HEIGHT = REGION_HEIGHT;
	this->LANDSAT_SCENE_ID = LANDSAT_SCENE_ID;
	this->FILE_DATE = FILE_DATE;
	this->YEAR = YEAR;
	this->MONTH = MONTH;
	this->DAY = DAY;
	this->HOUR = HOUR;
	this->MINUTE = MINUTE;
	this->SECONDS = SECONDS;
	this->LAND_AVG_TEMPERATURE = LAND_AVG_TEMPERATURE;
	this->SPACECRAFT_ID = SPACECRAFT_ID;
	this->SENSOR_ID = SENSOR_ID;
	this->WRS_PATH = WRS_PATH;
	this->WRS_ROW = WRS_ROW;
	this->FILE_NAME_BAND_1 = FILE_NAME_BAND_1;
	this->FILE_NAME_BAND_2 = FILE_NAME_BAND_2;
	this->FILE_NAME_BAND_3 = FILE_NAME_BAND_3;
	this->FILE_NAME_BAND_4 = FILE_NAME_BAND_4;
	this->FILE_NAME_BAND_5 = FILE_NAME_BAND_5;
	this->FILE_NAME_BAND_6 = FILE_NAME_BAND_6;
	this->FILE_NAME_BAND_7 = FILE_NAME_BAND_7;
	this->FILE_NAME_BAND_8 = FILE_NAME_BAND_8;
	this->FILE_NAME_BAND_9 = FILE_NAME_BAND_9;
	this->FILE_NAME_BAND_10 = FILE_NAME_BAND_10;
	this->FILE_NAME_BAND_11 = FILE_NAME_BAND_11;
	this->FILE_NAME_BAND_QUALITY = FILE_NAME_BAND_QUALITY;
	this->METADATA_FILE_NAME = METADATA_FILE_NAME;

	SCALE = (max(this->REGION_WIDTH - 1, this->REGION_HEIGHT - 1) * SCALE_EOFF);
}

SatDataCloud::~SatDataCloud(void)
{
	if (band2Data == NULL)
		delete[] band2Data;

	if (band3Data == NULL)
		delete[] band3Data;

	if (band4Data != NULL)
		delete[] band4Data;

	if (band5Data != NULL)
		delete[] band5Data;

	if (band6Data != NULL)
		delete[] band6Data;

	if (band7Data != NULL)
		delete[] band7Data;

	if (ir1Data != NULL)
		delete[] ir1Data;

	if (band2RegionData != NULL)
		delete[] band2RegionData;

	if (band3RegionData != NULL)
		delete[] band3RegionData;

	if (Band4RegionData != NULL)
		delete[]  Band4RegionData;

	if (Band5RegionData != NULL)
		delete[]  Band5RegionData;

	if (Band6RegionData != NULL)
		delete[]  Band6RegionData;

	if (Band7RegionData != NULL)
		delete[]   Band7RegionData;

	if (ir1RegionData != NULL)
		delete[] ir1RegionData;

	if (longitudeList != NULL)
		delete[] longitudeList;
	if (latitudeList != NULL)
		delete[] latitudeList;

	if (altitudeTable != NULL)
		delete[] altitudeTable;

	if (satZenithList != NULL)
		delete[] satZenithList;

	if (satAzimuthList != NULL)
		delete[] satAzimuthList;

	if (sunZenithAzimuthList != NULL)
		delete[] sunZenithAzimuthList;

	if (pixelTypeList != NULL)
		delete[] pixelTypeList;


	if (pixelTypeList2 != NULL)
		delete[] pixelTypeList2;

	if (pixelCloudIDList != NULL)
		delete[] pixelCloudIDList;

	if (shadowMaskList != NULL)
		delete[] shadowMaskList;

	if (eachCloudShadowMaskList != NULL)
		delete[] eachCloudShadowMaskList;


	if (cbhList != NULL)
		delete[] cbhList;

	if (cbhList_each_cloud != NULL)
		delete[] cbhList_each_cloud;

	if (cthList != NULL)
		delete[] cthList;

	if (cloudshadowPtList != NULL)
		delete[] cloudshadowPtList;


	if (geo_thick_data != NULL)
		delete[] geo_thick_data;

	if (extinction_data != NULL)
		delete[] extinction_data;


	if (effectiveRadius_data != NULL)
		delete[] effectiveRadius_data;

	if (Band4_thick_data != NULL)
		delete[]  Band4_thick_data;

	if (Band7_thick_data != NULL)
		delete[]  Band7_thick_data;

	if (vertexList != NULL)
		delete[] vertexList;

	if (edgeList != NULL)
		delete[] edgeList;

	if (faceList != NULL)
		delete[] faceList;

	if (isBaseBoundaryVertexList != NULL)
		delete[] isBaseBoundaryVertexList;

	//runIfoOut.close();
	//robustTestFile.close();
}

void SatDataCloud::Go(string input_path, string output_path, string basemesh_folder, string cloudmesh_folder)
{
	this->input_path = input_path;
	this->output_path = output_path;
	this->basemesh_folder = basemesh_folder;
	this->cloudmesh_folder = cloudmesh_folder;

	Init();
	Modeling();
	//ExportSatCloudIfo();
	//Export_PixelCloudIDList_CBH_CTH();
}

void SatDataCloud::Init(void)
{
	//runIfoOut.open("output\\runtime_info.txt");
	//if (!runIfoOut)
	//{
	//	cout << "文件不存在！";
	//	abort();
	//}
	//robustTestFile.open("output\\robustInfo.txt");
	//if (!robustTestFile)
	//{
	//	cout << "文件不存在！";
	//	abort();
	//}

	band2Data = NULL;
	band3Data = NULL;
	band4Data = NULL;
	band5Data = NULL;
	band6Data = NULL;
	band7Data = NULL;
	ir1Data = NULL;         //K

	//region data
	band2RegionData = NULL; //ratio
	band3RegionData = NULL; //ratio
	band3Data = NULL;       //ratio
	Band4RegionData = NULL; //ratio
	Band5RegionData = NULL;
	Band6RegionData = NULL;
	Band7RegionData = NULL; //K	
	ir1RegionData = NULL;   //K

	//longitude and latitude 
	longitudeList = NULL;
	latitudeList = NULL;

	//altitude
	altitudeTable = NULL;

	//satellite angles
	satAzimuthList = NULL;
	satZenithList = NULL;
	//solar angles
	sunZenithAzimuthList = NULL;
	//cloud mask
	pixelTypeList = NULL;
	pixelCloudIDList = NULL;
	pixelTypeList2 = NULL;

	//shadow mask
	shadowMaskList = NULL;
	eachCloudShadowMaskList = NULL;

	//clear-sky temperature
	clearSkyTemperatureList = NULL;

	//cloud bottom height
	cbhList = NULL;
	cbhList_each_cloud = NULL;
	cloudshadowPtList = NULL;

	//cloud top height
	cthList = NULL;
	//cloud geometric thickness
	geo_thick_data = NULL;
	//cloud parameters
	effectiveRadius_data = NULL;
	Band4_thick_data = NULL;
	Band7_thick_data = NULL;
	extinction_data = NULL;

	cur_cloud_id = 0;
	vertexList = NULL;
	faceList = NULL;
	edgeList = NULL;
	isBaseBoundaryVertexList = NULL;

	for (int i = 0; i < 20; i++)
	{
		dataRange[i].x = MAXVAL;
		dataRange[i].y = -MAXVAL;
	}

	ReadSatData(Band2);
	CreateRegionData(Band2);
	DeleteSatData();//save memory

	ReadSatData(Band3);
	CreateRegionData(Band3);
	DeleteSatData();//save memory

	ReadSatData(Band4);
	CreateRegionData(Band4);
	DeleteSatData();//save memory

	ReadSatData(Band5);
	CreateRegionData(Band5);
	DeleteSatData();//save memory

	ReadSatData(Band6);
	CreateRegionData(Band6);
	DeleteSatData();//save memory

	ReadSatData(Band7);
	CreateRegionData(Band7);
	DeleteSatData();//save memory

	ReadSatData(IR1);
	CreateRegionData(IR1);
	DeleteSatData();//save memory

	CreateLgLatLists();
	CreateAltitudeTable();
	CreateSatZenithAzimuthLists();
	CreateSunZenithAzimuth(YEAR, MONTH, DAY, HOUR, MINUTE, SECONDS);

	//CreateCloudMask(220);
	CreateCloudMask2();
	CreateCloudsBoudary();
	UpdateCloudMask();

	CreateShadowMask();
	//CreateCloudShadowPtList();

	CreateClearSkyTemperature();
}

bool SatDataCloud::ReadSatData(SatDataType band)
{
	CString bandFilename;

	width = REFLECTIVE_SAMPLES;
	height = REFLECTIVE_LINES;
	int*dataList = NULL;
	dataList = new int[width*height];

	switch (band)
	{
	case Band2:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_2).c_str();
		band2Data = dataList;
		break;

	case Band3:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_3).c_str();
		band3Data = dataList;
		break;

	case Band4:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_4).c_str();
		band4Data = dataList;
		break;

	case  Band5:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_5).c_str();
		band5Data = dataList;
		break;

	case  Band6:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_6).c_str();
		band6Data = dataList;
		break;

	case  Band7:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_7).c_str();
		band7Data = dataList;
		break;
	case IR1:
		bandFilename = (this->input_path + "/" + LANDSAT_SCENE_ID + "/" + FILE_NAME_BAND_10).c_str();
		ir1Data = dataList;
		break;
	}

	char fileName[100];
	for (int i = 0; i < bandFilename.GetLength(); i++)
	{
		fileName[i] = bandFilename[i];
	}
	fileName[bandFilename.GetLength()] = 0;
	fileName[bandFilename.GetLength() - 1] = 't';
	fileName[bandFilename.GetLength() - 2] = 'a';
	fileName[bandFilename.GetLength() - 3] = 'd';

	cout << fileName << endl;
	FILE *pFile = fopen(fileName, "rb");

	assert(pFile != NULL);

	fread(dataList, sizeof(int), width*height, pFile);
	fclose(pFile);
	return true;
}

void SatDataCloud::Modeling()
{
	ComputeCloudProperties_MEA2();//optical thickness
	CreateShape(0);

	ModifySmallCloudTopHeight();
	ComputeGeoThick();
	SmoothExtinctionField(10, 3);  //remove the exception  at the boundary of a cloud

	CreateAllBaseMeshes();
	CreateAllCloudMeshes();
}

bool SatDataCloud::CreateRegionData(SatDataType band)
{
	float*dataList = NULL;
	dataList = new float[REGION_WIDTH*REGION_HEIGHT];
	int* radianceData = NULL;

	switch (band)
	{
	case Band2:
		band2RegionData = dataList;
		radianceData = band2Data;
		break;
	case Band3:
		band3RegionData = dataList;
		radianceData = band3Data;
		break;
	case  Band4:
		Band4RegionData = dataList;
		radianceData = band4Data;
		break;
	case  Band5:
		Band5RegionData = dataList;
		radianceData = band5Data;
		break;
	case  Band6:
		Band6RegionData = dataList;
		radianceData = band6Data;
		break;
	case  Band7:
		Band7RegionData = dataList;
		radianceData = band7Data;
		break;
	case IR1:
		ir1RegionData = dataList;
		radianceData = ir1Data;
		break;
	}

	if (TOPLEFT_X + REGION_WIDTH > width || TOPLEFT_Y + REGION_HEIGHT > height)
	{
		cout << "Region Set Error!" << endl;
		return false;
	}

	// http://landsat.usgs.gov/Landsat8_Using_Product.php
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			int DN = radianceData[(i + TOPLEFT_Y)*width + j + TOPLEFT_X];

			switch (band)
			{
			case Band2:
				dataList[i*REGION_WIDTH + j] = DN*REFLECTANCE_MULT_BAND_2 + REFLECTANCE_ADD_BAND_2;
				dataList[i*REGION_WIDTH + j] /= cosf((90 - SUN_ELEVATION)*M_PI / 180);
				dataList[i*REGION_WIDTH + j] = min(1.0, (double)max((float)0, dataList[i*REGION_WIDTH + j]));
			case Band3:
				dataList[i*REGION_WIDTH + j] = DN*REFLECTANCE_MULT_BAND_3 + REFLECTANCE_ADD_BAND_3;
				dataList[i*REGION_WIDTH + j] /= cosf((90 - SUN_ELEVATION)*M_PI / 180);
				dataList[i*REGION_WIDTH + j] = min(1.0, (double)max((float)0, dataList[i*REGION_WIDTH + j]));
				break;
			case  Band4:
				dataList[i*REGION_WIDTH + j] = DN*REFLECTANCE_MULT_BAND_4 + REFLECTANCE_ADD_BAND_4;
				dataList[i*REGION_WIDTH + j] /= cosf((90 - SUN_ELEVATION)*M_PI / 180);
				dataList[i*REGION_WIDTH + j] = min(1.0, (double)max((float)0, dataList[i*REGION_WIDTH + j]));
				break;

			case  Band5:
				dataList[i*REGION_WIDTH + j] = DN*REFLECTANCE_MULT_BAND_5 + REFLECTANCE_ADD_BAND_5;
				dataList[i*REGION_WIDTH + j] /= cosf((90 - SUN_ELEVATION)*M_PI / 180);
				dataList[i*REGION_WIDTH + j] = min(1.0, (double)max((float)0, dataList[i*REGION_WIDTH + j]));
				break;
			case  Band6:
				dataList[i*REGION_WIDTH + j] = DN*REFLECTANCE_MULT_BAND_5 + REFLECTANCE_ADD_BAND_6;
				dataList[i*REGION_WIDTH + j] /= cosf((90 - SUN_ELEVATION)*M_PI / 180);
				dataList[i*REGION_WIDTH + j] = min(1.0, (double)max((float)0, dataList[i*REGION_WIDTH + j]));
				break;
			case  Band7:
				dataList[i*REGION_WIDTH + j] = DN*REFLECTANCE_MULT_BAND_7 + REFLECTANCE_ADD_BAND_7;
				dataList[i*REGION_WIDTH + j] /= cosf((90 - SUN_ELEVATION)*M_PI / 180);
				dataList[i*REGION_WIDTH + j] = min(1.0, (double)max((float)0, dataList[i*REGION_WIDTH + j]));
				break;
			case IR1:
				dataList[i*REGION_WIDTH + j] = DN*RADIANCE_MULT_BAND_10 + RADIANCE_ADD_BAND_10;
				dataList[i*REGION_WIDTH + j] = K2_CONSTANT_BAND_10 / logf(K1_CONSTANT_BAND_10 / dataList[i*REGION_WIDTH + j] + 1);
				break;
			}
			dataRange[int(band)].x = min(dataRange[int(band)].x, dataList[i*REGION_WIDTH + j]);
			dataRange[int(band)].y = max(dataRange[int(band)].y, dataList[i*REGION_WIDTH + j]);
		}
}

void SatDataCloud::CreateLgLatLists()
{
	latitudeList = new float[REGION_WIDTH * REGION_HEIGHT];
	longitudeList = new float[REGION_HEIGHT * REGION_WIDTH];

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			int idy = i + TOPLEFT_Y;
			int idx = j + TOPLEFT_X;

			float a = idx / float(width - 1);
			float b = idy / float(height - 1);

			float lg0, lg1;
			lg0 = CORNER_UL_LON_PRODUCT * (1 - a) + CORNER_UR_LON_PRODUCT * a;
			lg1 = CORNER_LL_LON_PRODUCT * (1 - a) + CORNER_LR_LON_PRODUCT * a;
			longitudeList[i * REGION_WIDTH + j] = lg0 * (1 - b) + lg1 * b;

			float lat0, lat1;
			lat0 = CORNER_UL_LAT_PRODUCT * (1 - a) + CORNER_UR_LAT_PRODUCT * a;
			lat1 = CORNER_LL_LAT_PRODUCT * (1 - a) + CORNER_LR_LAT_PRODUCT * a;
			latitudeList[i * REGION_WIDTH + j] = lat0 * (1 - b) + lat1 * b;
		}
}

void SatDataCloud::CreateSatZenithAzimuthLists()
{
	satZenithList = new float[REGION_WIDTH * REGION_HEIGHT];
	satAzimuthList = new float[REGION_WIDTH * REGION_HEIGHT];
	//assume the orbit is a straight line  y=kx+b where x,y are longitude and latitude respectively.

	k = tanf(M_PI / 2 - SAT_ORBIT_DIRECTION_ANGLE * M_PI / 180);

	float center_Lg = (CORNER_UL_LON_PRODUCT + CORNER_UR_LON_PRODUCT + CORNER_LL_LON_PRODUCT + CORNER_LR_LON_PRODUCT) / 4;
	float center_Lat = (CORNER_UL_LAT_PRODUCT + CORNER_UR_LAT_PRODUCT + CORNER_LL_LAT_PRODUCT + CORNER_LR_LAT_PRODUCT) / 4;

	b = center_Lat - k * center_Lg;

	float* subPoint_LatList = new float[REGION_WIDTH * REGION_HEIGHT];
	float* subPoint_LgList = new float[REGION_WIDTH * REGION_HEIGHT];

	float A = k;
	float B = -1;
	float C = b;

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			float lg = longitudeList[i * REGION_WIDTH + j];
			float lat = latitudeList[i * REGION_WIDTH + j];
			float distance = fabs(k * lg + b - lat) / sqrtf(1 + k * k);
			if (distance < F_ZERO)
			{
				subPoint_LatList[i * REGION_WIDTH + j] = lat;
				subPoint_LgList[i * REGION_WIDTH + j] = lg;
			}
			else
			{
				subPoint_LgList[i * REGION_WIDTH + j] = (B * B * lg - A * B * lat - A * C) / (A * A + B * B);
				subPoint_LatList[i * REGION_WIDTH + j] = (-B * A * lg + A * A * lat - B * C) / (A * A + B * B);
			}
		}
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			float lg = longitudeList[i * REGION_WIDTH + j] * M_PI / 180;
			float lat = latitudeList[i * REGION_WIDTH + j] * M_PI / 180;
			float sub_lat = subPoint_LatList[i * REGION_WIDTH + j] * M_PI / 180;
			float sub_lg = subPoint_LgList[i * REGION_WIDTH + j] * M_PI / 180;
			float distance = sqrtf(powf(lg - sub_lg, 2) + powf(lat - sub_lat, 2));
			if (distance < 0.0001)
			{
				satZenithList[i * REGION_WIDTH + j] = 0.0;
				satAzimuthList[i * REGION_WIDTH + j] = 90.0;
				continue;
			}
			Vector3 re = Vector3(EARTH_RADIUS * cosf(lat) * cosf(lg), EARTH_RADIUS * cosf(lat) * sinf(lg), EARTH_RADIUS * sinf(lat));
			Vector3 rs = Vector3((EARTH_RADIUS + SAT_ALTITUDE) * cosf(sub_lat) * cosf(sub_lg), (EARTH_RADIUS + SAT_ALTITUDE) * cosf(sub_lat) * sinf(sub_lg), (EARTH_RADIUS + SAT_ALTITUDE) * sinf(sub_lat));

			Vector3 rd = rs - re;
			float zenith = acosf(Dot(re, rd) / (Magnitude(re) * Magnitude(rd))) * 180 / M_PI;
			satZenithList[i * REGION_WIDTH + j] = zenith;

			Vector3 rn = Vector3(-sinf(lat) * cosf(lg), -sinf(lat) * sinf(lg), cosf(lat));
			Vector3 rh = rd - re * (Magnitude(rd) / Magnitude(re) * (cosf(zenith)));

			Vector3  East = Cross(rn, re / Magnitude(re));
			float azimuth = acosf(Dot(Normalize(rh), Normalize(rn))) * 180 / M_PI;
			/*			if(Dot(rh,East)<0)
							azimuth=360-azimuth;*/
			if (Dot(rh, East) < 0)
				azimuth = -azimuth;
			satAzimuthList[i * REGION_WIDTH + j] = azimuth;
		}

	delete[] subPoint_LgList;
	delete[] subPoint_LatList;
}

int SatDataCloud::CreateSunZenithAzimuth(int year, int month, int day, int hour, int minute, int second)
{
	extern vector<double> SunPosition(double year, double month, double day, double hour, double minute, double second,
		double height, double width, double offset, vector<double>& LongitudeLatitudeTable, vector<double>& AltitudeTable);

	vector<double> tempLongLatTable;
	tempLongLatTable.push_back(CORNER_UL_LON_PRODUCT);
	tempLongLatTable.push_back(CORNER_UL_LAT_PRODUCT);
	tempLongLatTable.push_back(CORNER_UR_LON_PRODUCT);
	tempLongLatTable.push_back(CORNER_UR_LAT_PRODUCT);
	tempLongLatTable.push_back(CORNER_LL_LON_PRODUCT);
	tempLongLatTable.push_back(CORNER_LL_LAT_PRODUCT);
	tempLongLatTable.push_back(CORNER_LR_LON_PRODUCT);
	tempLongLatTable.push_back(CORNER_LR_LAT_PRODUCT);

	vector<double> altitudeTable = { 0,0,0,0 };

	int width = 4;
	int height = 1;
	int offset = 0;

	vector<double> tempSunZenithAzimuthTable = SunPosition(year, month, day, hour, minute, second, height, width, offset, tempLongLatTable, altitudeTable);

	float CORNER_UL_ZENITH_PRODUCT = tempSunZenithAzimuthTable[0];
	float CORNER_UL_AZIMUTH_PRODUCT = tempSunZenithAzimuthTable[1];
	float CORNER_UR_ZENITH_PRODUCT = tempSunZenithAzimuthTable[2];
	float CORNER_UR_AZIMUTH_PRODUCT = tempSunZenithAzimuthTable[3];
	float CORNER_LL_ZENITH_PRODUCT = tempSunZenithAzimuthTable[4];
	float CORNER_LL_AZIMUTH_PRODUCT = tempSunZenithAzimuthTable[5];
	float CORNER_LR_ZENITH_PRODUCT = tempSunZenithAzimuthTable[6];
	float CORNER_LR_AZIMUTH_PRODUCT = tempSunZenithAzimuthTable[7];

	sunZenithAzimuthList = new float[2 * REGION_WIDTH*REGION_HEIGHT];
	for (int i = 0; i < REGION_HEIGHT; i++)
	{
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			int idy = i + TOPLEFT_Y;
			int idx = j + TOPLEFT_X;

			float a = idx / float(this->width - 1);
			float b = idy / float(this->height - 1);

			float zenith0, zenith1;
			zenith0 = CORNER_UL_ZENITH_PRODUCT*(1 - a) + CORNER_UR_ZENITH_PRODUCT*a;
			zenith1 = CORNER_LL_ZENITH_PRODUCT*(1 - a) + CORNER_LR_ZENITH_PRODUCT*a;
			sunZenithAzimuthList[2 * (i*REGION_WIDTH + j) + 0] = zenith0*(1 - b) + zenith1*b;

			float azimuth0, azimuth1;
			azimuth0 = CORNER_UL_AZIMUTH_PRODUCT*(1 - a) + CORNER_UR_AZIMUTH_PRODUCT*a;
			azimuth1 = CORNER_LL_AZIMUTH_PRODUCT*(1 - a) + CORNER_LR_AZIMUTH_PRODUCT*a;
			sunZenithAzimuthList[2 * (i*REGION_WIDTH + j) + 1] = azimuth0*(1 - b) + azimuth1*b;
			
			//cout << sunZenithAzimuthList[2 * (i*REGION_WIDTH + j)] << "   -    " << sunZenithAzimuthList[2 * (i*REGION_WIDTH + j) + 1] << endl;
		}
	}
	float zenith_center;
	zenith_center = (CORNER_UL_ZENITH_PRODUCT + CORNER_UR_ZENITH_PRODUCT + CORNER_LL_ZENITH_PRODUCT + CORNER_LR_ZENITH_PRODUCT) / 4;

	float azimuth_center;
	azimuth_center = (CORNER_UL_AZIMUTH_PRODUCT + CORNER_UR_AZIMUTH_PRODUCT + CORNER_LL_AZIMUTH_PRODUCT + CORNER_LR_AZIMUTH_PRODUCT) / 4;

	//PrintRunIfo("CreateSunZenithAzimuth");
	return 1;
}

void SatDataCloud::CreateAltitudeTable()
{
	//PrintRunIfo("CreateAltitudeTable");
	altitudeTable = new float[REGION_WIDTH*REGION_HEIGHT];
	for (int i = 0; i < REGION_WIDTH*REGION_HEIGHT; i++)
		altitudeTable[i] = 0;
}

void SatDataCloud::ComputeTriangleNormal( float normal[3], float PA[3],float PB[3],float PC[3] )
{

	float vecAB[3],vecAC[3];
	for(int i=0;i<3;i++)
	{
		vecAB[i]=PB[i]-PA[i];
		vecAC[i]=PC[i]-PA[i];
	}

	normal[0]=vecAB[1]*vecAC[2]-vecAB[2]*vecAC[1];
	normal[1]=vecAB[2]*vecAC[0]-vecAB[0]*vecAC[2];
	normal[2]=vecAB[0]*vecAC[1]-vecAB[1]*vecAC[0];

	float len=sqrtf(normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2]);
	for(int i=0;i<3;i++)
	{
		normal[i]/=len;
	}

}

void SatDataCloud::ComputeCloudProperties_MEA2()
{
	//refIdx1 : Imaginary part of the refractive index for liquid water wavelength1
	//refIdx2: Imaginary part of the refractive index for liquid water wavelength2

	float refIdx1 = 1.64e-8;
	float refIdx2 = 2.89e-4;

	//A1: Ground surface albedo in wavelength1
	//A2: Ground surface albedo in wavelength2
	float A1 = 0.0;
	float A2 = 0.0;

	float wavelength1 = Band4_WAVELENGTH*1.0e6;
	float wavelength2 = Band7_WAVELENGTH*1.0e6;

	Band4_thick_data = new float[REGION_WIDTH*REGION_HEIGHT];
	Band7_thick_data = new float[REGION_WIDTH*REGION_HEIGHT];
	effectiveRadius_data = new float[REGION_WIDTH*REGION_HEIGHT];

	struct  ReflectanceSampleIfo
	{
		float refValue;
		float thick1;
		float thick2;
		float radius;
	};

	int sampleRes = 30;
	ReflectanceSampleIfo* reflectanceSampleList = new ReflectanceSampleIfo[sampleRes];

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			if (pixelTypeList[i*REGION_WIDTH + j] == 0)
			{
				Band4_thick_data[i*REGION_WIDTH + j] = 0;
				Band7_thick_data[i*REGION_WIDTH + j] = 0;
				effectiveRadius_data[i*REGION_WIDTH + j] = 0;
				// cout<<"Raidus: "<< reflectanceSampleList[minSample].radius<<endl;
			}
			if (pixelTypeList[i*REGION_WIDTH + j] == 1)
			{
				// a vector of the effective radius  from 3 μm to 30 μm
				float MaxRadius = 35;
				float MinRadius = 5;
				float delta_radius = (MaxRadius - MinRadius) / (sampleRes - 1);

				float u = fabs(cos(satZenithList[REGION_WIDTH*i + j] * M_PI / 180));
				float u0 = cos(sunZenithAzimuthList[2 * (i*REGION_WIDTH + j) + 0] * M_PI / 180);
				float phi = satAzimuthList[REGION_WIDTH*i + j] - sunZenithAzimuthList[2 * (i*REGION_WIDTH + j) + 1];

				float k1 = 2 * M_PI / wavelength1;
				float k2 = 2 * M_PI / wavelength2;
				float kk1 = 4 * M_PI*refIdx1 / wavelength1;
				float kk2 = 4 * M_PI*refIdx2 / wavelength2;
				//escape function
				float K0u = 3.0 / 7 * (1 + 2 * u);
				float K0u0 = 3.0 / 7 * (1 + 2 * u0);

				for (int sample = 0; sample < sampleRes; sample++)
				{
					float curEfficientRadius = MinRadius + sample*delta_radius;
					//asymmetry parameter   0.85 um and 2.1um
					//asymmetry parameter   
					float  g1 = 1 - (0.12 + 0.5*powf((k1*curEfficientRadius), -2.0 / 3) - 0.15*kk1*curEfficientRadius);
					float  g2 = 1 - (0.12 + 0.5*powf((k2*curEfficientRadius), -2.0 / 3) - 0.15*kk2*curEfficientRadius);

					//single scattering albedo
					float ka = 5 * M_PI*refIdx2*(1 - kk2*curEfficientRadius) / wavelength2*(1 + 0.34*(1 - expf(-8 * wavelength2 / curEfficientRadius)));
					float ke = 1.5 / curEfficientRadius*(1 + 1.1 / powf((k2*curEfficientRadius), 2.0 / 3));
					float  w01 = 1.0; //no absorbing for vis channel 
					float  w02 = 1 - ka / ke;


					//Calculate scattering angle θ
					float theta = acos(-u*u0 + sin(sunZenithAzimuthList[2 * (i*REGION_WIDTH + j) + 0] * M_PI / 180)*sin(satZenithList[REGION_WIDTH*i + j] * M_PI / 180)*cos(phi*M_PI / 180));

					//Calculate Henyey-Greenstein phase function from g1
					float phase1 = (1 - g1*g1) / powf((1 + g1*g1 - 2 * g1*cos(theta)), 1.5);

					float A = 3.944;
					float B = -2.5;
					float C = 10.664;
					float RInf01 = (A + B*(u + u0) + C*u*u0 + phase1) / (4 * (u + u0));
					float  Band4R = Band4RegionData[i*REGION_WIDTH + j];
					float t1 = 1.0 / (K0u*K0u0 / (RInf01 - Band4RegionData[i*REGION_WIDTH + j]) - A1 / (1 - A1));
					//float t1=(RInf01- Band4Data[i*WIDTH+j])/(K0u*K0u0);
					float alpha = 1.07;
					float Thickness1 = (1.0 / t1 - alpha) / (0.75*(1 - g1));

					if (Thickness1 < 0)
					{
						cout << "water thickness too small" << endl;
						//runIfoOut << "water thickness too small" << endl;
						Band4_thick_data[i*REGION_WIDTH + j] = 0.1;
						Band7_thick_data[i*REGION_WIDTH + j] = 0.1;
						effectiveRadius_data[i*REGION_WIDTH + j] = 5;
						break;
					}
					if (Thickness1 > 100)
					{
						cout << "water thickness  too great!" << endl;
						//runIfoOut << "water thickness  too great!" << endl;
						Band4_thick_data[i*REGION_WIDTH + j] = 100;
						Band7_thick_data[i*REGION_WIDTH + j] = 100;
						effectiveRadius_data[i*REGION_WIDTH + j] = 15;
						break;
					}

					float xi1 = 2 * M_PI*curEfficientRadius / wavelength1;
					float xi2 = 2 * M_PI*curEfficientRadius / wavelength2;
					float Thickness2 = Thickness1*powf(wavelength2 / wavelength1, 2.0 / 3)*(1.1 + powf(xi2, 2.0 / 3)) / (1.1 + powf(xi1, 2.0 / 3));

					float x2 = sqrtf(3 * (1 - g2)*(1 - w02))*Thickness2;
					float y2 = 4 * sqrtf((1 - w02) / (3 * (1 - g2)));
					float tc = sinh(y2) / sinh(alpha*y2 + x2);
					float Delta = (4.86 - 13.08*u*u0 + 12.76*u*u*u0*u0)*expf(x2) / powf(Thickness2, 3);
					float t2 = tc;//-Delta;
					float a2 = expf(-y2) - tc*expf(-x2 - y2);

					float phase2 = (1 - g2*g2) / powf((1 + g2*g2 - 2 * g2*cos(theta)), 1.5);
					float RInf02 = (A + B*(u + u0) + C*u*u0 + phase2) / (4 * (u + u0));

					float mu = K0u0*K0u / RInf02;
					mu = mu*(1 - 0.05*y2);
					float curReflectance = RInf02*expf(-y2*mu) - (expf(-x2 - y2) - t2*A2 / (1 - A2*a2))*t2*K0u*K0u0;

					reflectanceSampleList[sample].refValue = curReflectance;
					reflectanceSampleList[sample].radius = curEfficientRadius;
					reflectanceSampleList[sample].thick1 = Thickness1;
					reflectanceSampleList[sample].thick2 = Thickness2;
				}

				float minDif = 9999;
				int minSample = 0;
				for (int sample = 0; sample < sampleRes; sample++)
				{
					float  Band4R = Band4RegionData[i*REGION_WIDTH + j];
					float irR = Band7RegionData[i*REGION_WIDTH + j];
					float sampleIrR = reflectanceSampleList[sample].refValue;
					float dif = fabs(irR - sampleIrR);
					if (dif < minDif)
					{
						minDif = dif;
						minSample = sample;
					}
				}
				Band4_thick_data[i*REGION_WIDTH + j] = reflectanceSampleList[minSample].thick1;
				Band7_thick_data[i*REGION_WIDTH + j] = reflectanceSampleList[minSample].thick2;
				effectiveRadius_data[i*REGION_WIDTH + j] = reflectanceSampleList[minSample].radius;
			}
		}
	delete[] reflectanceSampleList;
}

void SatDataCloud::ComputeGeoThick()
{
	//PrintRunIfo("Geometric thickness");

	geo_thick_data = new float[REGION_WIDTH*REGION_HEIGHT];
	extinction_data = new float[REGION_WIDTH*REGION_HEIGHT];

	float max_ext = -MAXVAL;
	float min_ext = MAXVAL;
	float max_thick = -MAXVAL;
	float min_thick = MAXVAL;
	float max_opticalthick = -MAXVAL;
	float min_opticalthick = MAXVAL;

	int exception_H_count = 0;

	for (int i = 0; i < REGION_HEIGHT; i++)
	{
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			extinction_data[i*REGION_WIDTH + j] = 0;
			geo_thick_data[i*REGION_WIDTH + j] = 0;

			if (pixelTypeList[i*REGION_WIDTH + j] == 1)  //water cloud
			{
				float rho = 1.0e3;
				float re = effectiveRadius_data[i*REGION_WIDTH + j] * 1.0e-6;
				float LWP = 2.0 / 3 * Band4_thick_data[i*REGION_WIDTH + j] * rho*re;

				float deltaZ = cthList[i*REGION_WIDTH + j] - cbhList[i*REGION_WIDTH + j];
				geo_thick_data[i*REGION_WIDTH + j] = deltaZ;

				if (deltaZ < 0)
				{
					deltaZ = 0;
					exception_H_count++;
				}
				float opticalthickness = Band4_thick_data[i*REGION_WIDTH + j];
				opticalthickness = min((float)100, max((float)0, opticalthickness));
				float beta = opticalthickness / (deltaZ + 30);

				extinction_data[i*REGION_WIDTH + j] = beta;
				//cout<<"Thickness:  "<<deltaZ<<endl;
			}
		}
	}
	for (int i = 0; i < REGION_HEIGHT; i++)
	{
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			if (pixelTypeList[i*REGION_WIDTH + j] == 1)  //cloud
			{
				max_thick = max(max_thick, geo_thick_data[i*REGION_WIDTH + j]);
				min_thick = min(min_thick, geo_thick_data[i*REGION_WIDTH + j]);

				max_ext = max(max_ext, extinction_data[i*REGION_WIDTH + j]);
				min_ext = min(min_ext, extinction_data[i*REGION_WIDTH + j]);

				max_opticalthick = max(max_opticalthick, Band4_thick_data[i*REGION_WIDTH + j]);
				min_opticalthick = min(min_opticalthick, Band4_thick_data[i*REGION_WIDTH + j]);
			}
		}
	}

	cout << "optical thick (min, max):  " << min_opticalthick << "    " << max_opticalthick << endl;
	cout << "thick (min, max):  " << min_thick << "    " << max_thick << endl;
	cout << "extinction (min, max):  " << min_ext << "    " << max_ext << endl;
	cout << "Exception thickness count:  " << exception_H_count << endl;

	//runIfoOut << "optical thick (min, max):  " << min_opticalthick << "    " << max_opticalthick << endl;
	//runIfoOut << "thick (min, max):  " << min_thick << "    " << max_thick << endl;
	//runIfoOut << "extinction (min, max):  " << min_ext << "    " << max_ext << endl;
	//runIfoOut << "Exception thickness count:  " << exception_H_count << endl;
}

void SatDataCloud::SmoothHeightField( float* heightField, int smooth_number,int smooth_size )
{

	if(heightField==NULL)
		return;

	float maxHeight=-MAXVAL;
	for(int i=0;i<REGION_HEIGHT;i++)
		for(int j=0;j<REGION_WIDTH;j++)
		{
		     maxHeight=max(maxHeight,heightField[i*REGION_WIDTH+j]);

		}

	IplImage *image= cvCreateImage(cvSize(REGION_WIDTH,REGION_HEIGHT), IPL_DEPTH_32F,1);     
	for(int i=0;i<REGION_HEIGHT;i++)
		for(int j=0;j<REGION_WIDTH;j++)
		{
			CvScalar cs;
			cs.val[0]=heightField[i*REGION_WIDTH+j]/maxHeight;
			cvSet2D(image,i,j,cs);

		}

		int smooth_count= smooth_number;
		while(smooth_count>0)
		{
			cvSmooth(image,image,CV_GAUSSIAN,smooth_size,smooth_size);
			smooth_count--;
		}


		for(int i=0;i<REGION_HEIGHT;i++)
			for(int j=0;j<REGION_WIDTH;j++)
			{
				CvScalar cs=cvGet2D(image,i,j);

				heightField[i*REGION_WIDTH+j]=float(cs.val[0]*maxHeight);

			}
			cvReleaseImage(&image);

}

void SatDataCloud::CreateCloudsBoudary()
{
	for (int i = 0; i < cloudsBoundary.size(); i++)
		cloudsBoundary[i].clear();
	cloudsBoundary.clear();

	int cloudContour_Number;
	CvSeq* cloudContourList = NULL;

	IplImage* image = cvCreateImage(cvSize(REGION_WIDTH, REGION_HEIGHT), IPL_DEPTH_8U, 1);
	for (int i = 0; i < REGION_HEIGHT; i++)
	{
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			CvScalar cs0;
			cs0.val[0] = pixelTypeList[i*REGION_WIDTH + j] * 255;
			cvSet2D(image, i, j, cs0);
		}
	}
	IplImage* dst = cvCreateImage(cvGetSize(image), 8, 3);

	CvMemStorage* storage = cvCreateMemStorage(0);
	cvThreshold(image, image, 125, 255, CV_THRESH_BINARY);   // 二值化  
	//cvNamedWindow("Source", 1);  
	//cvShowImage("Source", image);  
	// 提取轮廓  
	cloudContour_Number = cvFindContours(image, storage, &cloudContourList, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	cvZero(dst);        // 清空数组  
	CvSeq *contour = cloudContourList;
	double maxarea = 0;
	double minarea = 50;
	for (; contour != 0; contour = contour->h_next)
	{
		bool isRemove = false;

		double tmparea = fabs(cvContourArea(contour));
		if (tmparea < minarea)
		{
			cvSeqRemove(contour, 0); // 删除面积小于设定值的轮廓  
			cloudContour_Number--;
			isRemove = true;
			continue;
		}
		maxarea = max(maxarea, tmparea);

		// 创建一个色彩值  
		CvScalar color = CV_RGB(0, 255, 255);

		cvDrawContours(dst, contour, color, color, -1, 1, 8);   //绘制外部和内部的轮廓  

		if (!isRemove)
		{
			int onetourlength = contour->total;

			float area = float(fabs(cvContourArea(contour)));
			cloudsArea.push_back(area);

			vector<POINT>  points;
			points.clear();

			CvSeqReader reader;
			CvPoint pt = cvPoint(0, 0);
			cvStartReadSeq(contour, &reader);
			//开始提取
			for (int i = 0; i < onetourlength; i++)
			{
				CV_READ_SEQ_ELEM(pt, reader);
				POINT tempPt;
				tempPt.x = pt.x;
				tempPt.y = pt.y;

				points.push_back(tempPt);
			}
			cloudsBoundary.push_back(points);
		}
	}
	printf("Number of contours:  %d\n", cloudContour_Number);
	/*	cvNamedWindow("Components", 1);
		cvShowImage("Components", dst);
		cvWaitKey(0);
		cvDestroyWindow("Source");

		cvDestroyWindow("Components");  */

	cvReleaseImage(&dst);
	cvReleaseImage(&image);

	//normalize the list of the cloud areas
	float minArea = MAXVAL;
	float maxArea = -MAXVAL;
	for (int i = 0; i < cloudsArea.size(); i++)
	{
		minArea = min(minArea, cloudsArea[i]);
		maxArea = max(maxArea, cloudsArea[i]);
	}

	for (int i = 0; i < cloudsArea.size(); i++)
	{
		cloudsArea[i] = (cloudsArea[i] - minArea) / (maxArea - minArea);
	}

	//delete loop points

	for (int cloud_id = 0; cloud_id < cloudsBoundary.size(); cloud_id++)
	{
		int  pt_count = cloudsBoundary[cloud_id].size();
		int  Search_Number = pt_count / 3;
		bool *   isDeleteList = new bool[pt_count];
		for (int i = 0; i < pt_count; i++)
		{
			isDeleteList[i] = false;
		}
		for (int i = 0; i < pt_count; i++)
		{
			if (isDeleteList[i])
				continue;
			POINT  pt;
			pt.x = (cloudsBoundary[cloud_id][i]).x;
			pt.y = (cloudsBoundary[cloud_id][i]).y;

			vector<int> searchIDList;
			searchIDList.clear();

			for (int j = 1; j <= Search_Number; j++)
			{
				int id = (i + j) % pt_count;

				POINT  cur_pt;
				cur_pt.x = (cloudsBoundary[cloud_id][id]).x;
				cur_pt.y = (cloudsBoundary[cloud_id][id]).y;
				if (cur_pt.x == pt.x && cur_pt.y == pt.y)
				{
					for (int k = 0; k < searchIDList.size(); k++)
						isDeleteList[searchIDList[k]] = true;
					isDeleteList[id] = true;
				}
				else
				{
					searchIDList.push_back(id);
				}
			}
			searchIDList.clear();
		}
		int pixelId = 0;
		vector<POINT> ptlist;
		ptlist.clear();
		for (int i = 0; i < pt_count; i++)
		{
			if (!isDeleteList[i])
			{
				POINT pt = cloudsBoundary[cloud_id][i];
				ptlist.push_back(pt);
			}
		}
		cloudsBoundary[cloud_id].clear();
		cloudsBoundary[cloud_id] = ptlist;

		delete[] isDeleteList;
	}
	//label the cloud id for each pixel
	pixelCloudIDList = new int[REGION_WIDTH*REGION_HEIGHT];

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			pixelCloudIDList[i*REGION_WIDTH + j] = -1;

		}
	for (int cloud_id = 0; cloud_id < cloudsBoundary.size(); cloud_id++)
	{
		int number = cloudsBoundary[cloud_id].size();

		int min_x = 99999999;
		int max_x = -9999999;
		int min_y = 99999999;
		int max_y = -9999999;
		for (int k = 0; k < cloudsBoundary[cloud_id].size(); k++)
		{
			int x = cloudsBoundary[cloud_id][k].x;
			int y = cloudsBoundary[cloud_id][k].y;
			min_x = min(min_x, x);
			max_x = max(max_x, x);
			min_y = min(min_y, y);
			max_y = max(max_y, y);

			pixelCloudIDList[y*REGION_WIDTH + x] = cloud_id;
		}

		CvScalar scalar;
		CvMat* storage = cvCreateMat(1, number, CV_32FC2);
		for (int k = 0; k < cloudsBoundary[cloud_id].size(); k++)
		{
			scalar.val[0] = cloudsBoundary[cloud_id][k].x;
			scalar.val[1] = cloudsBoundary[cloud_id][k].y;
			cvSet1D(storage, k, scalar);
		}
		for (int i = min_y; i <= max_y; i++)
			for (int j = min_x; j <= max_x; j++)
			{
				CvPoint2D32f pt;
				pt.x = j;
				pt.y = i;
				if (cvPointPolygonTest(storage, pt, 1) > 0)
				{
					pixelCloudIDList[i*REGION_WIDTH + j] = cloud_id;
				}
			}
		cvReleaseMat(&storage);

		cout << cloud_id << endl;
		//runIfoOut << cloud_id << endl;
	}
	//for (int i = 0; i < REGION_WIDTH*REGION_HEIGHT; i++)
	//{
	//	if (pixelCloudIDList[i]!=-1)
	//	{
	//		cout << i<<"   "<<pixelCloudIDList[i] << endl;
	//	}
	//}
}

//注意：这里似乎存在非连通图的情况！
int SatDataCloud::CreateEachBaseMesh(int cur_cloud_id)
{
	int vertex_num = cloudsBoundary[cur_cloud_id].size();
	int img_maxWH = max(REGION_HEIGHT, REGION_WIDTH) - 1;

	float** points = new float*[2];
	points[0] = new float[vertex_num];
	points[1] = new float[vertex_num];

	for (int i = 0; i < vertex_num; i++)
	{
		float x = cloudsBoundary[cur_cloud_id][i].x*1.0 / img_maxWH;
		float y = (REGION_HEIGHT - 1 - cloudsBoundary[cur_cloud_id][i].y)*1.0 / img_maxWH;

		points[0][i] = x;
		points[1][i] = y;
	}

	extern void RefineMesh_Rev(float** points, int n, string output_filename);

	string fileName = this->output_path + "\\" + this->basemesh_folder + "\\basemesh";
	fileName += to_string(cur_cloud_id);
	fileName += ".off";
	RefineMesh_Rev(points, vertex_num, fileName);

	delete[] points[0];
	delete[] points[1];
	return 0;
}

int SatDataCloud::CreateAllBaseMeshes()
{

	for (int i = 0; i < cloudsBoundary.size(); i++)
	{
		int isOk = CreateEachBaseMesh(i);
		if (isOk < 0)
		{
			cout << "Create Base Mesh: " << i << " failed!" << endl;
			//runIfoOut << "Create Base Mesh: " << i << " failed!" << endl;
		}
		cout << "Base Mesh: " << i << endl;
		//runIfoOut << "Base Mesh: " << i << endl;
	}
	return 0;
}

int SatDataCloud::CreateAllCloudMeshes()
{
	for (int meshid = 0; meshid < cloudsBoundary.size(); meshid++)
	{
		//read base mesh
		string fileName = this->output_path + "\\" + this->basemesh_folder + "\\basemesh";
		fileName += to_string(meshid);
		fileName += ".off";
		ifstream in(fileName);
		if (!in)
		{
			cout << "文件不存在！";
			abort();
		}
		if (!in.fail())
		{
			char offStr[100];
			in >> offStr;
			in >> ver_number >> face_number >> edge_number;

			if (vertexList != NULL)
				delete[] vertexList;
			vertexList = new float[3 * ver_number];

			if (faceList != NULL)
				delete[] faceList;
			faceList = new int[3 * face_number];

			if (edgeList != NULL)
				delete[] edgeList;
			edgeList = new int[2 * edge_number];

			for (int i = 0; i < ver_number; i++)
			{
				in >> vertexList[3 * i + 0] >> vertexList[3 * i + 1] >> vertexList[3 * i + 2];
			}
			for (int i = 0; i < face_number; i++)
			{
				int verNumberEachFace = 3;
				in >> verNumberEachFace;

				in >> faceList[3 * i + 0] >> faceList[3 * i + 1] >> faceList[3 * i + 2];
			}
			for (int i = 0; i < edge_number; i++)
			{
				in >> edgeList[2 * i + 0] >> edgeList[2 * i + 1];
			}
			in.close();

			//interpolation geometry information to cloud mesh

			for (int i = 0; i < ver_number; i++)
			{
				float x = vertexList[3 * i + 0];
				float y = vertexList[3 * i + 1];
				float cth = InterPolateHeightField(cthList, x, y);

				vertexList[3 * i + 2] = max(cth, cbhList_each_cloud[meshid]) / SCALE;
			}
			//output cloudmesh to files

			string _filename = this->output_path + "\\" + this->cloudmesh_folder + "\\cloudmesh";
			_filename += to_string(meshid);
			_filename += ".off";
			ofstream out(_filename);
			if (!out)
			{
				cout << "文件不存在！";
				abort();
			}
			out << "OFF" << endl;
			out << ver_number << " " << face_number << " " << 0 << endl;
			for (int i = 0; i < ver_number; i++)
			{
				float x = vertexList[3 * i + 0];
				float y = vertexList[3 * i + 1];
				float z = vertexList[3 * i + 2];

				out << x << " " << y << " " << z << endl;
			}
			for (int i = 0; i < face_number; i++)
			{
				out << 3 << " " << faceList[3 * i + 0] << " " << faceList[3 * i + 1] << " " << faceList[3 * i + 2] << endl;
			}
			out.close();
			cout << "cloud mesh:  " << meshid << endl;
			//interpolation geometry information to cloud mesh
			for (int i = 0; i < ver_number; i++)
			{
				float x = vertexList[3 * i + 0];
				float y = vertexList[3 * i + 1];
				float extinction = InterPolateHeightField(extinction_data, x, y);
				//cout<<extinction<<endl;
				vertexList[3 * i + 2] = min(1.0, (double)max((float)0, extinction));
			}
			//output extinction_mesh to files

			string __filename = this->output_path + "\\" + this->cloudmesh_folder + "\\extinction_mesh";
			__filename += to_string(meshid);
			__filename += ".off";
			out.open(__filename);
			if (!out)
			{
				cout << "文件不存在！";
				abort();
			}
			out << "OFF" << endl;
			out << ver_number << " " << face_number << " " << 0 << endl;
			for (int i = 0; i < ver_number; i++)
			{
				float x = vertexList[3 * i + 0];
				float y = vertexList[3 * i + 1];
				float z = vertexList[3 * i + 2];

				out << x << " " << y << " " << z << endl;
			}
			for (int i = 0; i < face_number; i++)
			{
				out << 3 << " " << faceList[3 * i + 0] << " " << faceList[3 * i + 1] << " " << faceList[3 * i + 2] << endl;
			}
			out.close();
			cout << "extinction mesh:  " << meshid << endl;
			//runIfoOut << "extinction mesh:  " << meshid << endl;
		}
	}
	return 0;
}

float SatDataCloud::InterPolateHeightField(float* heightField, float x,float y )
{ 
	int img_maxWH=max(REGION_WIDTH, REGION_HEIGHT)-1;
	if(x<0||x>(REGION_WIDTH-1)/float(img_maxWH))
		return 0;
	if(y<0||y>(REGION_HEIGHT-1)/float(img_maxWH))
		return 0;

	float height=0.0;

	float interval_x=1.0/(img_maxWH);
	float interval_y=1.0/(img_maxWH);
	int x_index=int((x-0)/interval_x);
	float dx=(x-0)/interval_x-x_index;
	x_index=min(max(0,x_index),REGION_WIDTH-1);

	int y_index=int((y-0)/interval_y);
	float dy=(y-0)/interval_y-y_index;
	y_index=min(max(0,y_index),REGION_HEIGHT-1);

	if(x_index<REGION_WIDTH-1&&y_index<REGION_HEIGHT-1)
	{
		float a0=heightField[(REGION_HEIGHT-1-y_index)*REGION_WIDTH+x_index];
		float a1=heightField[(REGION_HEIGHT-1-y_index)*REGION_WIDTH+x_index+1];

		float b0=heightField[(REGION_HEIGHT-1-(y_index+1))*REGION_WIDTH+x_index];
		float b1=heightField[(REGION_HEIGHT-1-(y_index+1))*REGION_WIDTH+x_index+1];


		float a=a0*(1-dx)+a1*dx;
		float b=b0*(1-dx)+b1*dx;

		height=a*(1-dy)+b*dy;
	}

	if(x_index==REGION_WIDTH-1&&y_index<REGION_HEIGHT-1)
	{
		float a0=heightField[(REGION_HEIGHT-1-y_index)*REGION_WIDTH+x_index];
		float b0=heightField[(REGION_HEIGHT-1-(y_index+1))*REGION_WIDTH+x_index];

		height=a0*(1-dy)+b0*dy;
	}

	if(x_index<REGION_WIDTH-1&&y_index==REGION_HEIGHT-1)
	{
		float a0=heightField[(REGION_HEIGHT-1-y_index)*REGION_WIDTH+x_index];
		float a1=heightField[(REGION_HEIGHT-1-y_index)*REGION_WIDTH+x_index+1];

		height=a0*(1-dy)+a1*dy;
	}
	if(x_index==REGION_WIDTH-1&&y_index==REGION_HEIGHT-1)
	{
		float a0=heightField[(REGION_HEIGHT-1-y_index)*REGION_WIDTH+x_index];
		height=a0;
	}
	
	return height;
}

void SatDataCloud::CreateBaseBoundaryVertexList()
{
	if(isBaseBoundaryVertexList==NULL)
	{
	     isBaseBoundaryVertexList=new int[ver_number];
	}
	else
	{
		delete [] isBaseBoundaryVertexList;
		isBaseBoundaryVertexList=new int[ver_number];
	}

	for(int i=0;i<ver_number;i++)
	{
		isBaseBoundaryVertexList[i]=0;
	}

	for(int i=0;i<edge_number;i++)
	{

		int edgeInTriangle=0;

		int id1=edgeList[2*i+0];
		int id2=edgeList[2*i+1];


		for(int j=0;j<face_number;j++)
		{
			int idx=faceList[3*j+0];
			int idy=faceList[3*j+1];
			int idz=faceList[3*j+2];
			
			if(IsVertexOfTriangle(id1,idx,idy,idz) &&IsVertexOfTriangle(id2,idx,idy,idz))
				edgeInTriangle++;

		}

		if(edgeInTriangle==1)
		{
			isBaseBoundaryVertexList[edgeList[2*i+0]]=1;
			isBaseBoundaryVertexList[edgeList[2*i+1]]=1;
		}


	}


}

bool SatDataCloud::IsVertexOfTriangle(int id,int idx,int idy,int idz )
{

	if(id==idx || id==idy || id==idz)
		return true;
	else
		return false;

}

void SatDataCloud::CreateClearSkyTemperature()
{
	clearSkyTemperatureList = new float[REGION_WIDTH * REGION_HEIGHT];
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			clearSkyTemperatureList[i * REGION_WIDTH + j] = 300;

		}
	int* radiusList = new int[cloudsBoundary.size()];
	for (int cloud_id = 0; cloud_id < cloudsBoundary.size(); cloud_id++)
	{
		int min_x = 99999999;
		int max_x = -9999999;
		int min_y = 99999999;
		int max_y = -9999999;
		for (int k = 0; k < cloudsBoundary[cloud_id].size(); k++)
		{
			int x = cloudsBoundary[cloud_id][k].x;
			int y = cloudsBoundary[cloud_id][k].y;
			min_x = min(min_x, x);
			max_x = max(max_x, x);
			min_y = min(min_y, y);
			max_y = max(max_y, y);
		}
		radiusList[cloud_id] = int(max(max_x - min_x, max_y - min_y) * 0.75);

	}
	int max_radius = -1;
	for (int cloud_id = 0; cloud_id < cloudsBoundary.size(); cloud_id++)
	{
		max_radius = max(max_radius, radiusList[cloud_id]);

	}
	float* eachcloud_clearSkyTemperatureList = new float[cloudsBoundary.size()];

	//runIfoOut << "The temperature distribution: " << endl;
	for (int cloud_id = 0; cloud_id < cloudsBoundary.size(); cloud_id++)
	{
		int min_x = 99999999;
		int max_x = -9999999;
		int min_y = 99999999;
		int max_y = -9999999;
		for (int k = 0; k < cloudsBoundary[cloud_id].size(); k++)
		{
			int x = cloudsBoundary[cloud_id][k].x;
			int y = cloudsBoundary[cloud_id][k].y;
			min_x = min(min_x, x);
			max_x = max(max_x, x);
			min_y = min(min_y, y);
			max_y = max(max_y, y);
		}
		int center_x = (min_x + max_x) / 2;
		int center_y = (min_y + max_y) / 2;

		float radius = radiusList[cloud_id];//max_radius;
		float maxTem = -MAXVAL;
		float minTem = MAXVAL;
		float avg = 0.0;
		int count = 0;

		for (int idy = -radius; idy < radius; idy++)
			for (int idx = -radius; idx < radius; idx++)
			{
				int x = center_y + idx;
				int y = center_x + idy;
				if (x >= 0 && x < REGION_WIDTH && y >= 0 && y < REGION_HEIGHT)
				{
					if (pixelTypeList[y * REGION_WIDTH + x] == 0 && shadowMaskList[y * REGION_WIDTH + x] == 0)
					{
						maxTem = max(maxTem, ir1RegionData[y * REGION_WIDTH + x]);
						minTem = min(minTem, ir1RegionData[y * REGION_WIDTH + x]);
						avg += ir1RegionData[y * REGION_WIDTH + x];
						count++;
					}
				}
			}
		if (count == 0)
			eachcloud_clearSkyTemperatureList[cloud_id] = maxTem;
		else
			eachcloud_clearSkyTemperatureList[cloud_id] = avg / count;

		float derivation = 0.0;
		for (int idy = -radius; idy < radius; idy++)
			for (int idx = -radius; idx < radius; idx++)
			{
				int x = center_y + idx;
				int y = center_x + idy;
				if (x >= 0 && x < REGION_WIDTH && y >= 0 && y < REGION_HEIGHT)
				{
					if (pixelTypeList[y * REGION_WIDTH + x] == 0 && shadowMaskList[y * REGION_WIDTH + x] == 0)
					{
						derivation += powf(eachcloud_clearSkyTemperatureList[cloud_id] - ir1RegionData[y * REGION_WIDTH + x], 2.0);
					}
				}
			}
		if (count == 0)
			derivation = 0.0;
		else
			derivation = sqrtf(derivation / count);
		//runIfoOut << "Cloud  " << cloud_id << " " << "Tem:  " << eachcloud_clearSkyTemperatureList[cloud_id] << " maxTem: " << maxTem << "  minTem: " << minTem << "Derivation: " << derivation << endl;

		for (int i = 0; i < REGION_HEIGHT; i++)
			for (int j = 0; j < REGION_WIDTH; j++)
			{
				if (pixelCloudIDList[i * REGION_WIDTH + j] == cloud_id)
				{
					clearSkyTemperatureList[i * REGION_WIDTH + j] = eachcloud_clearSkyTemperatureList[cloud_id];
				}
			}
		cout << "clear sky Temperature of cloud:　" << cloud_id << endl;
		//runIfoOut << "clear sky Temperature of cloud:　" << cloud_id << endl;

	}
	delete[] radiusList;
	delete[] eachcloud_clearSkyTemperatureList;

	//string fileName = this->output_path + "\\band10.txt";
	//ofstream out(fileName);
	//for (int i = 0; i < REGION_HEIGHT; i++)
	//{
	//	for (int j = 0; j < REGION_WIDTH; j++)
	//	{
	//		out << ir1RegionData[i * REGION_WIDTH + j] << "    ";
	//	}
	//	out << endl;
	//}
	//out.close();
	//string _filename = this->output_path + "\\gt.txt";
	//out.open(_filename);
	//for (int i = 0; i < REGION_HEIGHT; i++)
	//{
	//	for (int j = 0; j < REGION_WIDTH; j++)
	//	{
	//		out << clearSkyTemperatureList[i * REGION_WIDTH + j] << "    ";
	//	}
	//	out << endl;
	//}
	//out.close();
}

void SatDataCloud::UpdateCloudMask()
{
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			pixelTypeList[i * REGION_WIDTH + j] = 0;

			if (pixelCloudIDList[i * REGION_WIDTH + j] >= 0)
			{
				pixelTypeList[i * REGION_WIDTH + j] = 1;
			}
		}
	string fileName = this->output_path + "\\" + this->cloudmesh_folder + "\\region_size.txt";
	ofstream outRegionSize(fileName);
	outRegionSize << cloudsBoundary.size() << endl;

	for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)
	{
		outRegionSize << contour_id << "  ";
		int max_x = -MAXVAL;
		int min_x = MAXVAL;
		int max_y = -MAXVAL;
		int min_y = MAXVAL;

		for (int i = 0; i < REGION_HEIGHT; i++)
			for (int j = 0; j < REGION_WIDTH; j++)
			{
				if (pixelCloudIDList[i * REGION_WIDTH + j] == contour_id)
				{
					max_x = max(max_x, j);
					min_x = min(min_x, j);
					max_y = max(max_y, i);
					min_y = min(min_y, i);
				}

			}
		outRegionSize << max_x - min_x << "   " << max_y - min_y << endl;
	}
}

void SatDataCloud::DeleteSatData()
{
	if(band2Data!=NULL)
		delete [] band2Data;
	if(band3Data!=NULL)
		delete [] band3Data;
	if(band4Data!=NULL)
		delete  [] band4Data;
	if(band5Data!=NULL)
		delete  [] band5Data;
	if(band6Data!=NULL)
		delete  [] band6Data;
	if(band7Data!=NULL)
		delete  [] band7Data;
	if(ir1Data!=NULL)
		delete [] ir1Data;

	band2Data=NULL;
	band3Data=NULL;
	band4Data=NULL;
	band5Data=NULL;
	band6Data=NULL;    
	band7Data=NULL;     	
	ir1Data=NULL;      //K
}

void SatDataCloud::SetCloudShadowPT( POINT cloudPt, POINT shadowPt,int cloud_id )
{
	cloudshadowPtList[2*cloud_id+0]=cloudPt;
	cloudshadowPtList[2*cloud_id+1]=shadowPt;

}

float SatDataCloud::PlanckFunction(float w,float T )
{
	float c=3.0e8;
	float h=6.626e-34;
	float k=1.3806e-23;

	float L=2*h*c*c/(pow(w,5)*(exp(h*c/(w*k*T))-1));

	return L;


}

float SatDataCloud::ComputeCloudTemperature( float ground_temp,float ir1_temp, float opticalThickness )
{
	float waveLength=10.9e-6;

	float  Trans=expf(-0.5*opticalThickness);
	float   cloud_radiance=(PlanckFunction(waveLength,ir1_temp)-Trans*PlanckFunction(waveLength,ground_temp))/(1-Trans);

	return  Radiance2Tempature(cloud_radiance);

	
}

float SatDataCloud::Radiance2Tempature( float radiance )
{

	float c=3.0e8;
	float h=6.626e-34;
	float k=1.3806e-23;
    float w=10.9e-6;

	float x1=2*h*c*c/radiance;
	float T=h*c/(logf(x1/pow(w,5)+1))/(w*k);

	return T;

}

void SatDataCloud::ComputeGeoThick_CTH_THICK()
{
	//PrintRunIfo("Geometric thickness");

	geo_thick_data = new float[REGION_WIDTH * REGION_HEIGHT];
	extinction_data = new float[REGION_WIDTH * REGION_HEIGHT];

	float max_ext = -MAXVAL;
	float min_ext = MAXVAL;
	float max_thick = -MAXVAL;
	float min_thick = MAXVAL;
	float max_opticalthick = -MAXVAL;
	float min_opticalthick = MAXVAL;

	int exception_H_count = 0;

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			extinction_data[i * REGION_WIDTH + j] = 0;
			geo_thick_data[i * REGION_WIDTH + j] = 0;

			if (pixelTypeList[i * REGION_WIDTH + j] == 1)  //water cloud
			{
				float  rho = 1.0e3;
				float re = effectiveRadius_data[i * REGION_WIDTH + j] * 1.0e-6;
				float LWP = 2.0 / 3 * Band4_thick_data[i * REGION_WIDTH + j] * rho * re;
				int N0 = 60e6;
				float alpha = 2;
				float  rn = re / (alpha + 2);
				float V = 4 / 3.0 * M_PI * N0 * pow(rn, 3) * 24;
				float LWC = rho * V;

				float cth = cthList[i * REGION_WIDTH + j];
				float deltaZ = fabs(LWP / LWC);

				if (deltaZ > 1500)
				{
					deltaZ = 1500;
				}
				if (deltaZ > cth)
				{
					//cout<<deltaZ<<" :  water Thick too greater! "<<endl;
					exception_H_count++;
				}
				float beta = Band4_thick_data[i * REGION_WIDTH + j] / deltaZ;

				extinction_data[i * REGION_WIDTH + j] = beta;
				geo_thick_data[i * REGION_WIDTH + j] = deltaZ;

				//cout<<"water deltaz:  "<<deltaZ<<endl;
			}
		}

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			if (pixelTypeList[i * REGION_WIDTH + j] == 1)  //cloud
			{
				max_thick = max(max_thick, geo_thick_data[i * REGION_WIDTH + j]);
				min_thick = min(min_thick, geo_thick_data[i * REGION_WIDTH + j]);

				max_ext = max(max_ext, extinction_data[i * REGION_WIDTH + j]);
				min_ext = min(min_ext, extinction_data[i * REGION_WIDTH + j]);

				max_opticalthick = max(max_opticalthick, Band4_thick_data[i * REGION_WIDTH + j]);
				min_opticalthick = min(min_opticalthick, Band4_thick_data[i * REGION_WIDTH + j]);
			}
		}

	cout << "optical thick (min, max):  " << min_opticalthick << "    " << max_opticalthick << endl;
	cout << "thick (min, max):  " << min_ext << "    " << max_thick << endl;
	cout << "extinction (min, max):  " << min_ext << "    " << max_ext << endl;
	cout << "Exception thickness count:  " << exception_H_count << endl;

	//runIfoOut << "optical thick (min, max):  " << min_opticalthick << "    " << max_opticalthick << endl;
	//runIfoOut << "thick (min, max):  " << min_ext << "    " << max_thick << endl;
	//runIfoOut << "extinction (min, max):  " << min_ext << "    " << max_ext << endl;
	//runIfoOut << "Exception thickness count:  " << exception_H_count << endl;

}

void SatDataCloud::CreateCloudBottomHeight_CTH_THICK()
{
	if(cthList==NULL||pixelTypeList==NULL||geo_thick_data==NULL)
		return;

	cbhList=new float[REGION_WIDTH*REGION_HEIGHT];

	for(int i=0;i<REGION_HEIGHT;i++)
		for(int j=0;j<REGION_WIDTH;j++)
		{
			if(pixelTypeList[i*REGION_WIDTH+j]==0)
			{
				cbhList[i*REGION_WIDTH+j]=0;

			}

			if( pixelTypeList[i*REGION_WIDTH+j]==1)
			{
				cbhList[i*REGION_WIDTH+j]=cthList[i*REGION_WIDTH+j]-geo_thick_data[i*REGION_WIDTH+j];
			}


		}

}

void SatDataCloud::CreateCloudTopHeight_CTH_THICK()
{
	if(ir1RegionData==NULL||pixelTypeList==NULL||clearSkyTemperatureList==NULL)
		return;

	if(cthList!=NULL)
		delete [] cthList;

	cthList=new float[REGION_WIDTH*REGION_HEIGHT];
	float minHeight=MAXVAL;
	float maxHeight=-MAXVAL;

	for(int i=0;i<REGION_HEIGHT;i++)
		for(int j=0;j<REGION_WIDTH;j++)
		{
			if(pixelTypeList[i*REGION_WIDTH+j]==0)
			{
				cthList[i*REGION_WIDTH+j]=0;
			}

			if( pixelTypeList[i*REGION_WIDTH+j]==1)
			{
			
				float cloudT= ir1RegionData[i*REGION_WIDTH+j];

				float cth=(clearSkyTemperatureList[i*REGION_WIDTH+j]-cloudT)*1000/6.48+altitudeTable[REGION_WIDTH*i+j];
				cthList[i*REGION_WIDTH+j]=min((float)5000,cth);

				minHeight=min(minHeight,cth);
				maxHeight=max(maxHeight,cth);

				//cout<<cth<<endl;
			}
		}
		cout<<"Height: (Min, Max):   " <<minHeight<<"  " <<maxHeight<<endl;
		//PrintRunIfo("CreateCloudTopHeight");
}

void SatDataCloud::SmoothExtinctionField( int smooth_number,int smooth_size )
{
	while(smooth_number)
	{
		SmoothHeightField(extinction_data,1,smooth_size);
		smooth_number--;
	}

	float min_ext=MAXVAL;
	float max_ext=-MAXVAL;
	for(int i=0;i<REGION_HEIGHT;i++)
		for(int j=0;j<REGION_WIDTH;j++)
		{
			if(pixelTypeList[i*REGION_WIDTH+j]==1)
			{
				max_ext=max(max_ext,extinction_data[i*REGION_WIDTH+j]);
				min_ext=min(min_ext,extinction_data[i*REGION_WIDTH+j]);
			}
		}
		cout<<"After Smooth: (min_ext,max_ext): "<<min_ext<<"   "<<max_ext<<endl;
		//runIfoOut<<"After Smooth: (min_ext,max_ext): "<<min_ext<<"   "<<max_ext<<endl;
}

//void SatDataCloud::ExportSatCloudIfo()
//{
//	IplImage* image = cvCreateImage(cvSize(REGION_WIDTH, REGION_HEIGHT), IPL_DEPTH_32F, 1);
//
//	//optical thickness
//	float min_value = MAXVAL;
//	float max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, Band4_thick_data[i * REGION_WIDTH + j]);
//				min_value = min(min_value, Band4_thick_data[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = Band4_thick_data[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	string fileName = this->output_path + "\\cloudifofile\\optical thickness.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//extinction
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, extinction_data[i * REGION_WIDTH + j]);
//				min_value = min(min_value, extinction_data[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = extinction_data[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//
//		}
//	fileName = this->output_path + "cloudifofile\\extinction.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//geometrical thickness
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, geo_thick_data[i * REGION_WIDTH + j]);
//				min_value = min(min_value, geo_thick_data[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = geo_thick_data[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//
//	fileName = this->output_path + "cloudifofile\\geometrical_thickness.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//effective radius
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, effectiveRadius_data[i * REGION_WIDTH + j]);
//				min_value = min(min_value, effectiveRadius_data[i * REGION_WIDTH + j]);
//			}
//
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = effectiveRadius_data[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	fileName = this->output_path + "cloudifofile\\effective_radius.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 2
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, band2RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, band2RegionData[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = band2RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	fileName = this->output_path + "cloudifofile\\band2.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 3
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, band3RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, band3RegionData[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = band3RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	fileName = this->output_path + "cloudifofile\\band3.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 4
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, Band4RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, Band4RegionData[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = Band4RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	fileName = this->output_path + "cloudifofile\\band4.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 5
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, Band5RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, Band5RegionData[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = Band5RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	fileName = this->output_path + "cloudifofile\\band5.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 6
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, Band6RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, Band6RegionData[i * REGION_WIDTH + j]);
//			}
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = Band6RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//		}
//	fileName = this->output_path + "cloudifofile\\band6.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 7
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, Band7RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, Band7RegionData[i * REGION_WIDTH + j]);
//			}
//
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = Band7RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//
//		}
//	fileName = this->output_path + "cloudifofile\\band7.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	//band 10
//	min_value = MAXVAL;
//	max_value = -MAXVAL;
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			if (pixelTypeList[i * REGION_WIDTH + j] == 1)
//			{
//				max_value = max(max_value, ir1RegionData[i * REGION_WIDTH + j]);
//				min_value = min(min_value, ir1RegionData[i * REGION_WIDTH + j]);
//
//			}
//
//		}
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = ir1RegionData[i * REGION_WIDTH + j] / max_value;
//			cvSet2D(image, i, j, cs0);
//
//		}
//	fileName = this->output_path + "cloudifofile\\band10.jpg";
//	cvSaveImage(fileName.c_str(), image);
//
//	IplImage* image2 = cvCreateImage(cvSize(REGION_WIDTH, REGION_HEIGHT), IPL_DEPTH_32F, 3);
//
//	//band 2,3,4 -> B,G,R
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = band2RegionData[i * REGION_WIDTH + j];
//			cs0.val[1] = band3RegionData[i * REGION_WIDTH + j];
//			cs0.val[2] = Band4RegionData[i * REGION_WIDTH + j];
//			cvSet2D(image2, i, j, cs0);
//
//		}
//	fileName = this->output_path + "cloudifofile\\rgb.jpg";
//	cvSaveImage(fileName.c_str(), image2);
//
//	//band 3,5,7 -> B,G,R
//	for (int i = 0; i < REGION_HEIGHT; i++)
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			CvScalar cs0;
//			cs0.val[0] = band3RegionData[i * REGION_WIDTH + j];
//			cs0.val[1] = Band5RegionData[i * REGION_WIDTH + j];
//			cs0.val[2] = Band7RegionData[i * REGION_WIDTH + j];
//			cvSet2D(image2, i, j, cs0);
//
//		}
//	fileName = this->output_path + "cloudifofile\\vrgb.jpg";
//	cvSaveImage("output\\cloudifofile\\vrgb.jpg", image2);
//	cvReleaseImage(&image2);
//}

void SatDataCloud::ModifySmallCloudTopHeight()
{
	     int ModifiedCloud=0;
		for(int contour_id=0;contour_id< cloudsBoundary.size(); contour_id++)    
		{    
			int max_x=-MAXVAL;
			int min_x=MAXVAL;
			int max_y=-MAXVAL;
			int min_y=MAXVAL;

			for(int i=0;i<REGION_HEIGHT;i++)
				for(int j=0;j<REGION_WIDTH;j++)
				{
					if(pixelCloudIDList[i*REGION_WIDTH+j]==contour_id)
					{
						max_x=max(max_x,j);
						min_x=min(min_x,j);
						max_y=max(max_y,i);
						min_y=min(min_y,i);

					}

				}

		    int  region_size=max(max_x-min_x,max_y-min_y);

			float  max_height=-MAXVAL;
			float  min_height=MAXVAL;

			int pixelCount=0;
			for(int i=0;i<REGION_HEIGHT;i++)
				for(int j=0;j<REGION_WIDTH;j++)
				{
					if(pixelCloudIDList[i*REGION_WIDTH+j]==contour_id)
					{
						max_height=max(cthList[i*REGION_WIDTH+j],max_height);
					    min_height=min(cthList[i*REGION_WIDTH+j],min_height);
						pixelCount++;

					}

				}


				float max_dis=-MAXVAL;
				for(int i=0;i<REGION_HEIGHT;i++)
					for(int j=0;j<REGION_WIDTH;j++)
					{
						if(pixelCloudIDList[i*REGION_WIDTH+j]==contour_id)
						{
							float  dis=MAXVAL;
							for(int id=0;id<cloudsBoundary[contour_id].size();id++)
							{
								POINT pt=cloudsBoundary[contour_id][id];
								float cur_dis=sqrtf((j-pt.x)*(j-pt.x)+(i-pt.y)*(i-pt.y));
								dis=min(cur_dis,dis);
								
							}
						  max_dis=max(max_dis,dis);

						}

					}


				int execeptionPixelCount=0;
				float max_thickness=max_height-min_height;
				if(max_thickness<region_size*30*0.8)
					max_thickness=region_size*30*0.5;

				for(int i=0;i<REGION_HEIGHT;i++)
					for(int j=0;j<REGION_WIDTH;j++)
					{
						if(pixelCloudIDList[i*REGION_WIDTH+j]==contour_id)
						{
							if(cthList[i*REGION_WIDTH+j]-min_height<0.0001)
							{
								//cout<<"   "<<cthList[i*REGION_WIDTH+j]<<endl;
								//runIfoOut<<"   "<<cthList[i*REGION_WIDTH+j]<<endl;

								float  dis=MAXVAL;
								for(int id=0;id<cloudsBoundary[contour_id].size();id++)
								{
									POINT pt=cloudsBoundary[contour_id][id];
									float cur_dis=sqrtf((j-pt.x)*(j-pt.x)+(i-pt.y)*(i-pt.y));
									dis=min(cur_dis,dis);

								}


								float er=effectiveRadius_data[i*REGION_WIDTH+j];
							    float LWP=2.0/3*Band4_thick_data[i*REGION_WIDTH+j]*er;
								float  thickness=LWP/0.58;
                                thickness=(max_thickness,thickness)*sqrtf(dis/max_dis);
								cthList[i*REGION_WIDTH+j]=cbhList[i*REGION_WIDTH+j]+thickness;

								execeptionPixelCount++;
							}

						}

					}


		}


		cout<<"The number of modified cloud:  "<<ModifiedCloud<<endl;

}

void SatDataCloud::CreateCloudMask2()
{
	if (pixelTypeList == NULL)
		pixelTypeList = new int[REGION_WIDTH * REGION_HEIGHT];
	if (pixelTypeList2 == NULL)
		pixelTypeList2 = new int[REGION_WIDTH * REGION_HEIGHT];

	int width;
	int height;
	width = REGION_WIDTH;
	height = REGION_HEIGHT;

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			pixelTypeList[i * width + j] = 0; //-1-background, 0-shdow,1-cloud
			pixelTypeList2[i * width + j] = 5; //1-cloud,2-Non-vegetated Lands,3-Snow/Ice,4-Water Bodies, 5-Vegetated Lands

			float R1 = band2RegionData[i * width + j];
			float R3 = Band4RegionData[i * width + j];
			float R4 = Band5RegionData[i * width + j];
			float R5 = Band6RegionData[i * width + j];

			if ((R1 < R3 && R3 < R4 && R4 < R5 * 1.07 && R5 < 0.65) || (R1 * 0.8 < R3 && R3 < R4 * 0.8 && R4 < R5 && R3 < 0.22))
			{
				pixelTypeList[i * width + j] = 0;//2;
				pixelTypeList2[i * width + j] = 2;
			}
			else
			{
				if ((R3 > 0.24 && R5<0.16 && R3>R4) || (R3 > 0.18 && R3 < 0.24 && R5<R3 - 0.08 && R3>R4))
				{
					pixelTypeList[i * width + j] = 0;//3;
					pixelTypeList2[i * width + j] = 3;
				}
				else
				{
					if ((R3 > R4 && R3 > R5 * 0.67 && R1 < 0.30 && R3 < 0.20) || (R3 > R4 * 0.8 && R3 > R5 * 0.67 && R3 < 0.06))
					{
						pixelTypeList[i * width + j] = 0;//4;
						pixelTypeList2[i * width + j] = 4;
					}
					else
					{
						if ((R1 > 0.15 || R3 > 0.18) && R5 > 0.12 && max(R1, R3) > R5 * 0.67)
						{
							pixelTypeList[i * width + j] = 1;
							pixelTypeList2[i * width + j] = 1;
						}
						else
						{
							pixelTypeList[i * width + j] = 0;//5;
							pixelTypeList2[i * width + j] = 5;
						}
					}
				}
			}
		}
	//other tests
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			float R1 = band2RegionData[i * width + j];
			float R3 = Band4RegionData[i * width + j];
			float R4 = Band5RegionData[i * width + j];
			float R5 = Band6RegionData[i * width + j];

			if (ir1RegionData[i * width + j] > 300)
			{
				pixelTypeList[i * width + j] = 0;
				pixelTypeList2[i * width + j] = 2;  //possible
			}
			if (R3 < 0.08)
			{
				pixelTypeList[i * width + j] = 0;
				pixelTypeList2[i * width + j] = 4;//possible
			}
		}

	//If a clear pixel surrounded by at least five cloudy neighboring pixels, it is reclassified as cloudy
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			if (pixelTypeList[i * width + j] != 1)//clear pixel
			{
				int cloudyCount = 0;
				for (int dy = -1; dy <= 1; dy++)
					for (int dx = -1; dx <= 1; dx++)
					{
						if (dx != 0 || dy != 0)
						{
							int idx = j + dx;
							int idy = i + dy;
							if (idx >= 0 && idx < REGION_WIDTH && idy >= 0 && idy < REGION_HEIGHT)
							{
								if (pixelTypeList[idy * REGION_WIDTH + idx] == 1)
								{
									cloudyCount++;
								}
							}
						}
					}

				if (cloudyCount >= 5)
				{
					pixelTypeList[i * width + j] = 1;
					pixelTypeList2[i * width + j] = 1;
				}
			}
		}
	//output 
	//output to file 
	CvSize size;
	size.width = width;
	size.height = height;
	IplImage* pImg_mask = cvCreateImage(size, IPL_DEPTH_8U, 1);

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			CvScalar cs0;
			int  temp = 0;
			cs0.val[0] = temp;
			switch (pixelTypeList[i * width + j])
			{
			case 1:
				temp = 255;
				break;
			case 2:
				temp = 128;
				break;
			case 3:
				temp = 200;
				break;
			case 4:
				temp = 32;
				break;
			case 5:
				temp = 64;
				break;
			default:
				temp = 0;
				break;
			}
			cs0.val[0] = temp;
			cvSet2D(pImg_mask, i, j, cs0);
		}

	string fileName = this->output_path + "\\CloudMask2.jpg";
	cvSaveImage(fileName.c_str(), pImg_mask);

	pImg_mask = cvLoadImage(fileName.c_str(), 0);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			CvScalar cs0;
			cs0 = cvGet2D(pImg_mask, i, j);
			int  temp = cs0.val[0];
			pixelTypeList[i * width + j] = 0;

			if (temp > 253)
				pixelTypeList[i * width + j] = 1;
			if (temp < 130 && temp>126)
				pixelTypeList[i * width + j] = 2;
			if (temp < 202 && temp>198)
				pixelTypeList[i * width + j] = 3;
			if (temp < 34 && temp>30)
				pixelTypeList[i * width + j] = 4;
			if (temp < 66 && temp>62)
				pixelTypeList[i * width + j] = 5;
		}
}

void SatDataCloud::CreateShadowMask()
{
	if (shadowMaskList == NULL)
		shadowMaskList = new int[REGION_WIDTH*REGION_HEIGHT];

	if (eachCloudShadowMaskList == NULL)
		eachCloudShadowMaskList = new int[REGION_WIDTH*REGION_HEIGHT];

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			shadowMaskList[i*REGION_WIDTH + j] = 0;
			eachCloudShadowMaskList[i*REGION_WIDTH + j] = -1;
		}

	// create potential or extended shadow mask
	float sunTheta = (90 - SUN_ELEVATION)*M_PI / 180.0;
	float sunPhi = SUN_AZIMUTH*M_PI / 180.0;

	Vector3 sunVec = Vector3(sinf(sunTheta)*sinf(sunPhi), sinf(sunTheta)*cosf(sunPhi), cosf(sunTheta));

	for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	{
		cout << "Projection:  " << contour_id << endl;
		//runIfoOut << "Projection:  " << contour_id << endl;

		float MIN_HEIGHT = 50;
		float MAX_HEIGHT = 10000;

		float min_tem = MAXVAL;
		float max_tem = -MAXVAL;

		for (int i = 0; i < REGION_HEIGHT; i++)
			for (int j = 0; j < REGION_WIDTH; j++)
			{
				if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
				{
					min_tem = min(min_tem, ir1RegionData[i*REGION_WIDTH + j]);
					max_tem = max(max_tem, ir1RegionData[i*REGION_WIDTH + j]);
				}
			}

		//max_tem=max(clearSkyTemperatureList[contour_id],min_tem);
		MAX_HEIGHT = min((double)MAX_HEIGHT, (max_tem - min_tem) / 5.0 * 1000);

		for (int i = 0; i < REGION_HEIGHT; i++)
			for (int j = 0; j < REGION_WIDTH; j++)
			{
				if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
				{
					for (float cloudheight = MIN_HEIGHT; cloudheight < MAX_HEIGHT; cloudheight += 30)
					{
						int shadow_x = -1;
						int shadow_y = -1;
						Project(shadow_x, shadow_y, j, i, cloudheight, sunVec);

						int  R = 1;
						for (int dy = -R; dy <= R; dy++)
							for (int dx = -R; dx <= R; dx++)
							{
								int idx = shadow_x + dx;
								int idy = shadow_y + dy;
								if (idx >= 0 && idx < REGION_WIDTH&&idy >= 0 && idy < REGION_HEIGHT  && pixelTypeList[idy*REGION_WIDTH + idx] == 0)
								{
									shadowMaskList[idy*REGION_WIDTH + idx] = 1;
									eachCloudShadowMaskList[idy*REGION_WIDTH + idx] = contour_id;
								}
							}
					}
				}
			}
	}
	//exclude the apparent clear sky pixels, see [Developing clear-sky, cloud and cloud shadow mask for producing clear-sky MODIS, Luo et al., 2008] for details
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			float B3 = band2RegionData[i*REGION_WIDTH + j];
			float B1 = Band4RegionData[i*REGION_WIDTH + j];
			float B2 = Band5RegionData[i*REGION_WIDTH + j];
			float B6 = Band6RegionData[i*REGION_WIDTH + j];

			if (shadowMaskList[i*REGION_WIDTH + j] == 1)
			{
				if (pixelTypeList2[i*REGION_WIDTH + j] != 3)  // Land and water
				{
					if (!(max(B2, B6) / B3 < 1.5 && B1 < 0.12 && B2 < 0.24 && B6 < 0.24))
					{
						shadowMaskList[i*REGION_WIDTH + j] = 0;
						eachCloudShadowMaskList[i*REGION_WIDTH + j] = -1;
					}
				}
				if (pixelTypeList2[i*REGION_WIDTH + j] == 3)  // snow/ice
				{
					if (!(B2 / B3 < 0.75))
					{
						shadowMaskList[i*REGION_WIDTH + j] = 0;
						eachCloudShadowMaskList[i*REGION_WIDTH + j] = -1;
					}
				}
			}
		}
	//output to file 
	//CvSize size;
	//size.width = REGION_WIDTH;
	//size.height = REGION_HEIGHT;
	//IplImage* pImg_mask = cvCreateImage(size, IPL_DEPTH_32F, 3);

	//Vector3* colorTable = new Vector3[cloudsBoundary.size()];
	//for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	//{
	//	float  R = (rand() % 255) / 255.0;
	//	float  G = (rand() % 255) / 255.0;
	//	float  B = (rand() % 255) / 255.0;
	//	colorTable[contour_id] = Vector3(R, G, B);
	//}
	//for (int i = 0; i < REGION_HEIGHT; i++)
	//	for (int j = 0; j < REGION_WIDTH; j++)
	//	{
	//		CvScalar cs0;
	//		if (pixelTypeList[i*REGION_WIDTH + j] == 1)
	//		{
	//			cs0.val[2] = 1.0;
	//			cs0.val[1] = 1.0;
	//			cs0.val[0] = 102 / 255.0;//1.0;
	//		}
	//		if (pixelTypeList[i*REGION_WIDTH + j] != 1)
	//		{
	//			cs0.val[2] = 130 / 255.0; //0.0;
	//			cs0.val[1] = 190 / 255.0;//0.7;
	//			cs0.val[0] = 102 / 255.0;//0.0;
	//		}
	//		if (eachCloudShadowMaskList[i*REGION_WIDTH + j] >= 0)
	//		{
	//			int cloud_id = eachCloudShadowMaskList[i*REGION_WIDTH + j];
	//			cs0.val[2] = 0.0;//colorTable[cloud_id].z;
	//			cs0.val[1] = 128 / 255.0;//colorTable[cloud_id].y;
	//			cs0.val[0] = 102 / 255.0;//colorTable[cloud_id].x;
	//		}
	//		cvSet2D(pImg_mask, i, j, cs0);
	//	}
	//cvSaveImage("output\\shadowmask_Color.jpg", pImg_mask);
	//delete[] colorTable;

	//pImg_mask = cvCreateImage(size, IPL_DEPTH_32F, 1);
	//for (int i = 0; i < REGION_HEIGHT; i++)
	//	for (int j = 0; j < REGION_WIDTH; j++)
	//	{
	//		CvScalar cs0;
	//		if (pixelTypeList[i*REGION_WIDTH + j] == 1)
	//		{
	//			cs0.val[0] = 1.0;
	//		}
	//		if (pixelTypeList[i*REGION_WIDTH + j] != 1)
	//		{
	//			cs0.val[0] = 0.5;
	//		}
	//		if (eachCloudShadowMaskList[i*REGION_WIDTH + j] >= 0)
	//		{
	//			cs0.val[0] = 0.0;
	//		}
	//		cvSet2D(pImg_mask, i, j, cs0);
	//	}
	//cvSaveImage("output\\shadowmask.jpg", pImg_mask);

	//ofstream outShadowMask("output\\shadow.txt");
	//ofstream outEachCloudShadowMaskList("output\\eachCloudShadow.txt");
	//for (int i = 0; i < REGION_HEIGHT; i++)
	//{
	//	for (int j = 0; j < REGION_WIDTH; j++)
	//	{
	//		outShadowMask << shadowMaskList[i*REGION_WIDTH + j] << "  ";
	//		outEachCloudShadowMaskList << eachCloudShadowMaskList[i*REGION_WIDTH + j] << "  " << endl;
	//	}
	//	outShadowMask << endl;
	//	outEachCloudShadowMaskList << endl;
	//}
	//outShadowMask.close();
	//outEachCloudShadowMaskList.close();

	//ofstream outCloudShadow("output\\cloud_shadow_ycq.txt");
	//for (int i = 0; i < REGION_HEIGHT; i++)
	//{
	//	for (int j = 0; j < REGION_WIDTH; j++)
	//	{
	//		int  label = -1;
	//		if (pixelCloudIDList[i*REGION_WIDTH + j] < 0 && eachCloudShadowMaskList[i*REGION_WIDTH + j] < 0)
	//			label = -1;
	//		if (pixelCloudIDList[i*REGION_WIDTH + j] >= 0)
	//			label = pixelCloudIDList[i*REGION_WIDTH + j];
	//		if (eachCloudShadowMaskList[i*REGION_WIDTH + j] >= 0)
	//		{
	//			label = eachCloudShadowMaskList[i*REGION_WIDTH + j] + cloudsBoundary.size();
	//		}
	//		outCloudShadow << label << "  ";
	//	}
	//	outCloudShadow << endl;
	//}
	//outCloudShadow.close();
}

void SatDataCloud::Project(int& shadow_x, int& shadow_y, int idx, int idy, float height, Vector3 sunVec)
{
	int maxWH = max(REGION_WIDTH, REGION_HEIGHT);
	Vector3 cloudPos = Vector3(float(idx) / maxWH, float(REGION_HEIGHT - 1 - idy) / maxWH, height / (maxWH * 30));
	float t = cloudPos.z / sunVec.z;
	float xs = cloudPos.x - t*sunVec.x;
	float ys = cloudPos.y - t*sunVec.y;
	shadow_x = int(xs*maxWH);
	shadow_y = int(REGION_HEIGHT - 1 - maxWH*ys);
}

void SatDataCloud::CreateShape(int specifiedCloud)
{
	if (pixelTypeList == NULL || pixelTypeList2 == NULL || shadowMaskList == NULL || clearSkyTemperatureList == NULL || ir1RegionData == NULL)
		return;

	if (cbhList_each_cloud != NULL)
		delete[] cbhList_each_cloud;
	cbhList_each_cloud = new float[cloudsBoundary.size()];
	for (int i = 0; i < cloudsBoundary.size(); i++)
	{
		cbhList_each_cloud[i] = 0;
	}

	if (cbhList != NULL)
		delete[]cbhList;
	cbhList = new float[REGION_WIDTH*REGION_HEIGHT];
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			cbhList[i*REGION_WIDTH + j] = 0;
		}

	if (cthList != NULL)
		delete[] cthList;

	cthList = new float[REGION_WIDTH*REGION_HEIGHT];

	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			cthList[i*REGION_WIDTH + j] = 0;
		}
	//cloud pixel temperature
	float* cloudTemList = new float[REGION_WIDTH*REGION_HEIGHT];
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			cloudTemList[i*REGION_WIDTH + j] = 0.0;
			if (pixelTypeList[i*REGION_WIDTH + j] == 1)
			{
				float ir1T = ir1RegionData[i*REGION_WIDTH + j];
				float  Band4_opticalThickness = Band4_thick_data[i*REGION_WIDTH + j];
				float cloudT = ComputeCloudTemperature(clearSkyTemperatureList[i*REGION_WIDTH + j], ir1T, Band4_opticalThickness);
				cloudTemList[i*REGION_WIDTH + j] = cloudT;
			}
		}

	//ofstream  outCTT("output\\avg_ctt.txt");
	//for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	//{
	//	float avg_tem = MAXVAL;
	//	int pixelNumber = 0;
	//	for (int i = 0; i < REGION_HEIGHT; i++)
	//		for (int j = 0; j < REGION_WIDTH; j++)
	//		{
	//			if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
	//			{
	//				avg_tem = min(avg_tem, cloudTemList[i*REGION_WIDTH + j]);
	//				pixelNumber++;
	//			}
	//		}
	//	/*	avg_tem/=pixelNumber;*/
	//	outCTT << contour_id << "  " << avg_tem << endl;
	//}
	//outCTT.close();

	//temp shadow mask
	bool* tempShadowMaskList = new bool[REGION_WIDTH*REGION_HEIGHT];
	for (int i = 0; i < REGION_HEIGHT; i++)
		for (int j = 0; j < REGION_WIDTH; j++)
		{
			tempShadowMaskList[i*REGION_WIDTH + j] = false;
		}
	float sunTheta = (90 - SUN_ELEVATION)*M_PI / 180.0;
	float sunPhi = SUN_AZIMUTH*M_PI / 180.0;
	Vector3 sunVec = Vector3(sinf(sunTheta)*sinf(sunPhi), sinf(sunTheta)*cosf(sunPhi), cosf(sunTheta));
	class Similarity
	{
	public:
		float similarityValue;
		float lapseRate1;  // exterior of cloud
		float lapseRate2; //in cloud
		float baseHeight;
		Similarity()
		{
			similarityValue = 0.0;
			lapseRate1 = 6.5;
			lapseRate2 = 6.5;
			baseHeight = 1000;

		}

		Similarity(const Similarity  &other)
		{
			similarityValue = other.similarityValue;
			lapseRate1 = other.lapseRate1;
			lapseRate2 = other.lapseRate2;
			baseHeight = other.baseHeight;

		}
		Similarity(float similarityValue, float lapseRate1, float lapseRate2, float baseHeight)
		{
			this->similarityValue = similarityValue;
			this->lapseRate1 = lapseRate1;
			this->lapseRate2 = lapseRate2;
			this->baseHeight = baseHeight;
		}
		Similarity operator +(const Similarity  &other)
		{
			Similarity sumSim;
			sumSim.similarityValue = this->similarityValue + other.similarityValue;
			sumSim.lapseRate1 = this->lapseRate1 + other.lapseRate1;
			sumSim.lapseRate2 = this->lapseRate2 + other.lapseRate2;
			sumSim.baseHeight = this->baseHeight + other.baseHeight;

			return sumSim;

		}
	};
	vector<vector<Similarity>> similarityVecVec;

	for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	{
		cout << "C" << contour_id << ": ";

		vector<Similarity> similarityVec;
		similarityVec.clear();

		double MIN_HEIGHT = 50;
		double MAX_HEIGHT = 10000;

		double min_tem = MAXVAL;
		double max_tem = -MAXVAL;

		for (int i = 0; i < REGION_HEIGHT; i++)
			for (int j = 0; j < REGION_WIDTH; j++)
			{
				//cout << pixelCloudIDList[i*REGION_WIDTH + j] << endl;
				if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
				{
					cout << "ir1RegionData  :" << ir1RegionData[i*REGION_WIDTH + j] << endl;
					min_tem = min(min_tem, (double)ir1RegionData[i*REGION_WIDTH + j]);
					max_tem = max(max_tem, (double)ir1RegionData[i*REGION_WIDTH + j]);
				}
			}

		max_tem = max((double)clearSkyTemperatureList[contour_id], min_tem);
		MAX_HEIGHT = min((double)MAX_HEIGHT, (double)((max_tem - min_tem) / 5.0 * 1000));

		for (float lapse_rate1 = 5.0; lapse_rate1 <= 10.0; lapse_rate1 += 0.1)
			//  for(float lapse_rate2=4.8;lapse_rate2<=10.0;lapse_rate2+=0.5)
			for (float baseHeight = MIN_HEIGHT; baseHeight < MAX_HEIGHT; baseHeight += 30)
			{
				float lapse_rate2 = lapse_rate1;

				for (int i = 0; i < REGION_HEIGHT; i++)
					for (int j = 0; j < REGION_WIDTH; j++)
					{
						tempShadowMaskList[i*REGION_WIDTH + j] = false;
					}

				for (int i = 0; i < REGION_HEIGHT; i++)
					for (int j = 0; j < REGION_WIDTH; j++)
					{
						if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
						{
							float cloudbaseT = clearSkyTemperatureList[i*REGION_WIDTH + j] - baseHeight / 1000 * lapse_rate1;
							float cloudT = cloudTemList[i*REGION_WIDTH + j];
							float topHeight = baseHeight + max((float)0.0, (cloudbaseT - cloudT)) / lapse_rate2 * 1000;

							for (float cloudHeight = baseHeight; cloudHeight <= topHeight; cloudHeight += 30)
							{
								int shadow_x = -1;
								int shadow_y = -1;
								Project(shadow_x, shadow_y, j, i, cloudHeight, sunVec);

								int  R = 0;
								for (int dy = -R; dy <= R; dy++)
									for (int dx = -R; dx <= R; dx++)
									{
										int idx = shadow_x + dx;
										int idy = shadow_y + dy;
										if (idx >= 0 && idx < REGION_WIDTH&&idy >= 0 && idy < REGION_HEIGHT  && pixelTypeList[idy*REGION_WIDTH + idx] == 0)
										{
											tempShadowMaskList[idy*REGION_WIDTH + idx] = 1;
										}
									}

							}//	for(float cloudHeight=baseHeight
						}//if(pixelCloudIDList
					}//for i,j

				int tempShadowCount = 0;
				int bothShadowCount = 0;
				int originalShadowCount = 0;

				for (int i = 0; i < REGION_HEIGHT; i++)
					for (int j = 0; j < REGION_WIDTH; j++)
					{
						if (tempShadowMaskList[i*REGION_WIDTH + j] == 1)
						{
							tempShadowCount++;
							if (shadowMaskList[i*REGION_WIDTH + j] == 1)
							{
								bothShadowCount++;
							}
						}
						if (eachCloudShadowMaskList[i*REGION_WIDTH + j] == contour_id)
						{
							originalShadowCount++;
						}
					}
				float similarityValue = 0.0;
				if (tempShadowCount > 0)
					similarityValue = bothShadowCount / float(tempShadowCount + originalShadowCount - bothShadowCount);

				Similarity  cur_Similarity(similarityValue, lapse_rate1, lapse_rate2, baseHeight);
				similarityVec.push_back(cur_Similarity);
			}//	for(float baseHeight=MIN_HEIGHT
		 //find the index of maximum similarity 
		int max_id = -1;
		float max_similartiy = -MAXVAL;
		int vec_size = similarityVec.size();
		for (int id = 0; id < similarityVec.size(); id++)
		{
			float cur_similarity = similarityVec[id].similarityValue;
			if (max_similartiy < similarityVec[id].similarityValue)
			{
				max_similartiy = similarityVec[id].similarityValue;
				max_id = id;
			}
		}
		int count = 0;
		vector<Similarity> used_similartiyVec;
		for (int id = 0; id < similarityVec.size(); id++)
		{
			if (fabs(max_similartiy - similarityVec[id].similarityValue) < F_ZERO)
			{
				count++;
				used_similartiyVec.push_back(similarityVec[id]);
			}
		}
		similarityVecVec.push_back(used_similartiyVec);
		cout << used_similartiyVec.size() << endl;
		//runIfoOut << used_similartiyVec.size() << endl;
		for (int id = 0; id < used_similartiyVec.size(); id++)
		{
			cout << "--" << id << " " << used_similartiyVec[id].baseHeight << " " << used_similartiyVec[id].lapseRate1 << "  " << used_similartiyVec[id].lapseRate2 << endl;
			//runIfoOut << "--" << id << " " << used_similartiyVec[id].baseHeight << " " << used_similartiyVec[id].lapseRate1 << "  " << used_similartiyVec[id].lapseRate2 << endl;
		}
	}//for(int contour_id

	Similarity*    finalSimilarityVec = new Similarity[cloudsBoundary.size()];
	bool* isFinishedList = new bool[cloudsBoundary.size()];
	Vector2* cloudCenterList = new Vector2[cloudsBoundary.size()];
	float* distanceBetweenClouds = new float[cloudsBoundary.size()*cloudsBoundary.size()];

	int min_ps_solution = MAXVAL;
	int min_ps_solution_cloud = -1;

	int finishedCount = 0;
	for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	{
		isFinishedList[contour_id] = false;
		vector<Similarity> similarityVec = similarityVecVec[contour_id];
		if (similarityVec.size() <= 2)
		{
			finishedCount++;
			isFinishedList[contour_id] = true;
			Similarity  finalSim = Similarity(0.0, 0.0, 0.0, 0.0);
			for (int solutionId = 0; solutionId < similarityVec.size(); solutionId++)
			{
				finalSim = finalSim + similarityVec[solutionId];
			}
			finalSim.baseHeight /= similarityVec.size();
			finalSim.lapseRate1 /= similarityVec.size();
			finalSim.lapseRate2 /= similarityVec.size();
			finalSim.similarityValue /= similarityVec.size();
			finalSimilarityVec[contour_id] = finalSim;
		}
		if (min_ps_solution > similarityVec.size())
		{
			min_ps_solution = similarityVec.size();
			min_ps_solution_cloud = contour_id;
		}

		Vector2 cur_center = Vector2(0, 0);
		for (int bid = 0; bid < cloudsBoundary[contour_id].size(); bid++)
		{
			POINT pt = cloudsBoundary[contour_id][bid];
			cur_center = cur_center + Vector2(pt.x, pt.y);

		}
		cur_center = cur_center / cloudsBoundary[contour_id].size();

		cloudCenterList[contour_id] = cur_center;
	}
	cout << "Total,  Finished:  " << cloudsBoundary.size() << "  " << finishedCount << endl;
	//runIfoOut << "Total,  Finished:  " << cloudsBoundary.size() << "  " << finishedCount << endl;
	if (min_ps_solution > 2)
	{
		cout << "!!!!!!!!!!!!!!!!!!!!!!!!" << "Note that there is no cloud with only one solution!" << endl;
		//runIfoOut << "!!!!!!!!!!!!!!!!!!!!!!!!" << "Note that there is no cloud with only one solution!" << endl;
		vector<Similarity> similarityVec = similarityVecVec[min_ps_solution_cloud];
		isFinishedList[min_ps_solution_cloud] = true;
		finalSimilarityVec[min_ps_solution_cloud] = similarityVec[0];
	}
	for (int loop = 0; loop < cloudsBoundary.size(); loop++)
	{
		int cloudNumber = cloudsBoundary.size();

		for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
		{
			distanceBetweenClouds[loop*cloudNumber + contour_id] = Magnitude(cloudCenterList[loop] - cloudCenterList[contour_id]);
		}
	}
	bool isDone = true;
	for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	{
		if (!isFinishedList[contour_id])
		{
			isDone = false;
		}
	}

	while (!isDone)
	{
		int unfinished_cloud = -1;
		int finished_cloud = -1;
		float min_dis = MAXVAL;

		for (int loop = 0; loop < cloudsBoundary.size(); loop++)
		{
			if (!isFinishedList[loop])
			{
				int cloudNumber = cloudsBoundary.size();
				float tempDis = MAXVAL;
				float temp_cloud = -1;
				for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
				{
					if (isFinishedList[contour_id] && tempDis > distanceBetweenClouds[loop*cloudNumber + contour_id])
					{
						tempDis = distanceBetweenClouds[loop*cloudNumber + contour_id];
						temp_cloud = contour_id;
					}
				}
				if (min_dis > tempDis)
				{
					unfinished_cloud = loop;
					min_dis = tempDis;
					finished_cloud = temp_cloud;
				}
			}
		}
		vector<Similarity>  cur_similarityVec = similarityVecVec[unfinished_cloud];
		Similarity optimal;
		float min_dif_base = MAXVAL;
		for (int vec_id = 0; vec_id < cur_similarityVec.size(); vec_id++)
		{
			float cur_dif_base = sqrt(powf(cur_similarityVec[vec_id].lapseRate1 - finalSimilarityVec[finished_cloud].lapseRate1, 2.0) + powf(cur_similarityVec[vec_id].lapseRate2 - finalSimilarityVec[finished_cloud].lapseRate2, 2.0));
			if (cur_dif_base < min_dif_base)
			{
				min_dif_base = cur_dif_base;
				optimal = cur_similarityVec[vec_id];
			}
		}
		finalSimilarityVec[unfinished_cloud] = optimal;
		isFinishedList[unfinished_cloud] = true;

		isDone = true;
		for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
		{
			if (!isFinishedList[contour_id])
			{
				isDone = false;
			}
		}
	}
	for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	{
		float GT = 300;
		bool isFind = false;
		for (int i = 0; i < REGION_HEIGHT && !isFind; i++)
			for (int j = 0; j < REGION_WIDTH && !isFind; j++)
			{
				if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
				{
					GT = clearSkyTemperatureList[i*REGION_WIDTH + j];
					isFind = true;

				}
			}
		Similarity cur_similarity = finalSimilarityVec[contour_id];
		float cbt = GT - cur_similarity.baseHeight / 1000.0*cur_similarity.lapseRate1;

		cout << "C" << contour_id << "  sim,lr1,lr2,cbh,cbt: " << cur_similarity.similarityValue << " " << cur_similarity.lapseRate1 << " " << cur_similarity.lapseRate2 << " " << cur_similarity.baseHeight << " " << cbt << endl;

		//runIfoOut << "C" << contour_id << "  sim,lr1,lr2,cbh,cbt: " << cur_similarity.similarityValue << " " << cur_similarity.lapseRate1 << " " << cur_similarity.lapseRate2 << " " << cur_similarity.baseHeight << " " << cbt << endl;

		if (contour_id == specifiedCloud)
		{
			//robustTestFile << cur_similarity.similarityValue << "  " << cbt << "  " << cur_similarity.baseHeight << "  " << cur_similarity.lapseRate1 << endl;
		}

		cbhList_each_cloud[contour_id] = cur_similarity.baseHeight;

		for (int i = 0; i < REGION_HEIGHT; i++)
			for (int j = 0; j < REGION_WIDTH; j++)
			{
				if (pixelCloudIDList[i*REGION_WIDTH + j] == contour_id)
				{
					float cloudbaseT = clearSkyTemperatureList[i*REGION_WIDTH + j] - cbhList_each_cloud[contour_id] / 1000.0*cur_similarity.lapseRate1;
					float cloudT = cloudTemList[i*REGION_WIDTH + j];
					float topHeight = cbhList_each_cloud[contour_id] + max((float)0.0, (cloudbaseT - cloudT)) / cur_similarity.lapseRate2 * 1000;
					cbhList[i*REGION_WIDTH + j] = cbhList_each_cloud[contour_id];
					cthList[i*REGION_WIDTH + j] = topHeight;
				}
			}
	}

	delete[] tempShadowMaskList;
	delete[] finalSimilarityVec;
	delete[] distanceBetweenClouds;
	delete[] cloudCenterList;
	delete[] isFinishedList;
	delete[] cloudTemList;

	//ofstream outBaseHeight("output\\cbh.txt");
	//for (int contour_id = 0; contour_id < cloudsBoundary.size(); contour_id++)    //for each cloud
	//{
	//	outBaseHeight << contour_id << "  " << cbhList_each_cloud[contour_id] << endl;
	//}
}

//void SatDataCloud::Export_PixelCloudIDList_CBH_CTH()
//{
//	ofstream  out("output\\cbh_robust.txt");
//	if (cbhList_each_cloud != NULL)
//	{
//		for (int i = 0; i < cloudsBoundary.size(); i++)
//		{
//			out << i << "     " << cbhList_each_cloud[i] << endl;
//		}
//	}
//	out.close();
//	out.open("output\\cth_robust.txt");
//	for (int i = 0; i < REGION_HEIGHT; i++)
//	{
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			out << cthList[i*REGION_WIDTH + j] << "  ";
//		}
//		out << endl;
//	}
//	out.close();
//
//	out.open("output\\pixelCloudIDlist_robust.txt");
//	for (int i = 0; i < REGION_HEIGHT; i++)
//	{
//		for (int j = 0; j < REGION_WIDTH; j++)
//		{
//			out << pixelCloudIDList[i*REGION_WIDTH + j] << "  ";
//		}
//		out << endl;
//	}
//	out.close();
//}

bool SatDataCloud::isNeighborListExistShadow(int cur_cloud_id, int* shadowMask, int* orginEachCloudShadowList, int j, int i)
{
	if (i + 1 < REGION_HEIGHT)
	{
		if (shadowMask[(i + 1)*REGION_WIDTH + j] > 0 && orginEachCloudShadowList[(i + 1)*REGION_WIDTH + j] == cur_cloud_id)
		{
			return true;
		}
	}
	if (i - 1 >= 0)
	{
		if (shadowMask[(i - 1)*REGION_WIDTH + j] > 0 && orginEachCloudShadowList[(i - 1)*REGION_WIDTH + j] == cur_cloud_id)
		{
			return true;
		}
	}

	if (j + 1 < REGION_WIDTH)
	{
		if (shadowMask[i*REGION_WIDTH + j + 1] > 0 && orginEachCloudShadowList[i*REGION_WIDTH + j + 1] == cur_cloud_id)
		{
			return true;
		}
	}
	if (j - 1 >= 0)
	{
		if (shadowMask[i*REGION_WIDTH + j - 1] > 0 && orginEachCloudShadowList[i*REGION_WIDTH + j - 1] == cur_cloud_id)
		{
			return true;
		}
	}
	return false;
}

bool SatDataCloud::isNeighborListExistBackground(int* shadowMask, int* pixelCloudIDList, int j, int i)
{
	if (i + 1 < REGION_HEIGHT)
	{
		if (shadowMask[(i + 1)*REGION_WIDTH + j] == 0 && pixelCloudIDList[(i + 1)*REGION_WIDTH + j] == -1)
		{
			return true;
		}
	}
	if (i - 1 >= 0)
	{
		if (shadowMask[(i - 1)*REGION_WIDTH + j] == 0 && pixelCloudIDList[(i - 1)*REGION_WIDTH + j] == -1)
		{
			return true;
		}
	}
	if (j + 1 < REGION_WIDTH)
	{
		if (shadowMask[i*REGION_WIDTH + j + 1] == 0 && pixelCloudIDList[i*REGION_WIDTH + j + 1] == -1)
		{
			return true;
		}
	}
	if (j - 1 >= 0)
	{
		if (shadowMask[i*REGION_WIDTH + j - 1] == 0 && pixelCloudIDList[i*REGION_WIDTH + j - 1] == -1)
		{
			return true;
		}
	}
	return false;
}
