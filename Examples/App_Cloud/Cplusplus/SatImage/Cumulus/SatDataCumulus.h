#pragma once
#include "global.h"
#include "Utility/Vector.h"
#include "Utility/Color.h"
#include "Utility/perlin.h"
#include <vector>
#include <algorithm> 
#include "string"

enum SatDataType{Band2,Band3, Band4, Band5,Band6, Band7,IR1,RGB,VRGB};
enum DrawType {LgLat,SatelliteZenithAzimuth,SolarZenithAzimuth,PixelType,ClearSkyTemperature,CloudsBoundary,
CloudHeightField,GeoThick,OpticalThick,EffectiveRadius,Extinction,BaseMesh,CloudMesh};

class SatDataCloud
{
public:
	SatDataCloud(int TOPLEFT_X = 3561,
		int TOPLEFT_Y = 2063,
		int REGION_WIDTH = 650,
		int REGION_HEIGHT = 650,
		double SCALE_EOFF = 30.0,
		string LANDSAT_SCENE_ID = "LC81160482013179LGN01",
		long long FILE_DATE = 20130628235849,
		int YEAR = 2013,
		int MONTH = 6,
		int DAY = 28,
		int HOUR = 2,
		int MINUTE = 18,
		int SECONDS = 35,
		int LAND_AVG_TEMPERATURE = 283,
		string SPACECRAFT_ID = "LANDSAT_8",
		string SENSOR_ID = "OLI_TIRS",
		int WRS_PATH = 116,
		int WRS_ROW = 48,
		string FILE_NAME_BAND_1 = "LC81160482013179LGN01_B1.TIF",
		string FILE_NAME_BAND_2 = "LC81160482013179LGN01_B2.TIF",
		string FILE_NAME_BAND_3 = "LC81160482013179LGN01_B3.TIF",
		string FILE_NAME_BAND_4 = "LC81160482013179LGN01_B4.TIF",
		string FILE_NAME_BAND_5 = "LC81160482013179LGN01_B5.TIF",
		string FILE_NAME_BAND_6 = "LC81160482013179LGN01_B6.TIF",
		string FILE_NAME_BAND_7 = "LC81160482013179LGN01_B7.TIF",
		string FILE_NAME_BAND_8 = "LC81160482013179LGN01_B8.TIF",
		string FILE_NAME_BAND_9 = "LC81160482013179LGN01_B9.TIF",
		string FILE_NAME_BAND_10 = "LC81160482013179LGN01_B10.TIF",
		string FILE_NAME_BAND_11 = "LC81160482013179LGN01_B11.TIF",
		string FILE_NAME_BAND_QUALITY = "LC81160482013179LGN01_BQA.TIF",
		string METADATA_FILE_NAME = "LC81160482013179LGN01_MTL.txt"
	);
	~SatDataCloud(void);

private:
	int TOPLEFT_X;
	int TOPLEFT_Y;
	int REGION_WIDTH;
	int REGION_HEIGHT;
	double SCALE;
	string LANDSAT_SCENE_ID;
	long long FILE_DATE;
	int YEAR;
	int MONTH;
	int DAY;
	int HOUR;
	int MINUTE;
	int SECONDS;
	int LAND_AVG_TEMPERATURE;
	string SPACECRAFT_ID;
	string SENSOR_ID;
	int WRS_PATH;
	int WRS_ROW;
	string FILE_NAME_BAND_1;
	string FILE_NAME_BAND_2;
	string FILE_NAME_BAND_3;
	string FILE_NAME_BAND_4;
	string FILE_NAME_BAND_5;
	string FILE_NAME_BAND_6;
	string FILE_NAME_BAND_7;
	string FILE_NAME_BAND_8;
	string FILE_NAME_BAND_9;
	string FILE_NAME_BAND_10;
	string FILE_NAME_BAND_11;
	string FILE_NAME_BAND_QUALITY;
	string METADATA_FILE_NAME;

	string input_path;
	string output_path;
	string basemesh_folder;
	string cloudmesh_folder;

private:
	ofstream runIfoOut;
	void Init();
	void Modeling();

	//input data
	int  width;
	int  height;
	int* band2Data; //band2, 0-65535
	int* band3Data; //band3, 0-65535
	int* band4Data; //band4, 0-65535
	int* band5Data; //band5, 0-65535
	int* band6Data; //band6, 0-65535
	int* band7Data; //band7, 0-65535
	int* ir1Data;   //band10,0-65535

	bool ReadSatData(SatDataType band);
	void DeleteSatData();

	//region data
	float* band2RegionData; //reflectance
	float* band3RegionData; //reflectance
	float* Band4RegionData; //reflectance
	float* Band5RegionData; //reflectance
	float* Band6RegionData; //reflectance
	float* Band7RegionData; //reflectance
	float* ir1RegionData;       //K
	Vector2 dataRange[20];   // [min,max] at each band
	bool CreateRegionData(SatDataType band);

	//longitude and latitude for each pixel within the region
	float* longitudeList;
	float* latitudeList;
	void CreateLgLatLists();

	//altitude for each pixel within the region
	float* altitudeTable;
	void CreateAltitudeTable();

	//Satellite Zenith and Azimuth for each pixel within the region
	float* satZenithList;
	float* satAzimuthList;
	void CreateSatZenithAzimuthLists();

	//orbit line y=kx+b;
	float k;
	float b;

	//the azimuth angle  and  the zenith angle of the sun
	float* sunZenithAzimuthList;
	int CreateSunZenithAzimuth(int year, int month, int day, int hour, int minute, int second);

	//cloud mask 
	int* pixelTypeList;  //-1-background, 0-shdow,1-cloud/1-cloud,2-Non-vegetated Lands,3-Snow/Ice,4-Water Bodies, 5-Vegetated Lands
	int* pixelTypeList2;  //1-cloud,2-Non-vegetated Lands,3-Snow/Ice,4-Water Bodies, 5-Vegetated Lands
	//void CreateCloudMask(float threshold);
	//CreateCloudMask2 : Implementation on Landsat Data of a simple Cloud-mask algorithm developed for MODIS Land Bands, Oreopoulos et al.2011
	void CreateCloudMask2();
	void UpdateCloudMask();
	//void ExportCloudMask();

	//clear-sky temperature
	float* clearSkyTemperatureList; //ir1
	void CreateClearSkyTemperature();

	//cloud mask
	vector<vector<POINT>> cloudsBoundary;
	vector<float> cloudsArea;
	int* pixelCloudIDList; //-1----non-cloud pixel, 0~clouds.boundary.size()-1----cloud id
	void CreateCloudsBoudary();

	//shadow mask
	int* shadowMaskList;
	int* eachCloudShadowMaskList;
	void CreateShadowMask();
	void Project(int& shadow_x, int& shadow_y, int idx, int idy, float height, Vector3 sunVec);

	bool isNeighborListExistShadow(int cur_cloud_id, int* shadowMask, int* orginEachCloudShadowList, int idx, int idy);
	bool isNeighborListExistBackground(int* shadowMask, int* pixelCloudIDList, int idx, int idy);

	//cloud properties
	float* Band4_thick_data; // optical thickness in VIS band
	float* Band7_thick_data; // optical thickness in NIR band
	float* effectiveRadius_data;  // effective radius
	void ComputeCloudProperties_MEA();  //the method of ycq
	void ComputeCloudProperties_MEA2(); //the method of MEA

	//estimated cloud base height and top height from the similarity between the projection of the cloud and the shadow
	float* cbhList;	//cloud bottom height
	float* cbhList_each_cloud;
	float* cthList;	//cloud top height	
	ofstream robustTestFile;
	void CreateShape(int specifiedCloud);//top and base height
	void ModifySmallCloudTopHeight();
	void SmoothHeightField(float* heightField, int smooth_number, int smooth_size);
	float PlanckFunction(float waveLength, float T);
	float Radiance2Tempature(float radiance);
	float ComputeCloudTemperature(float ground_temp, float ir1_temp, float opticalThickness);
	void ComputeTriangleNormal(float normal[3], float PA[3], float PB[3], float PC[3]);

	//geometric thickness
	float* geo_thick_data;          // thickness 
	float* extinction_data;          // extinction
	void ComputeGeoThick();
	void SmoothExtinctionField(int smooth_number, int smooth_size);

	//label the matched pixel pair on  the edge of  the shadow and the edge of the cloud
	POINT* cloudshadowPtList;
	//void CreateCloudShadowPtList();
	void SetCloudShadowPT(POINT cloudPt, POINT shadowPt, int cloud_id);
	//void ExportCloudShadowPTList();
	//void ExportCBH();

	//for the base mesh of each cloud
	int cur_cloud_id;
	float* vertexList;
	int* edgeList;
	int* faceList;
	int ver_number;
	int edge_number;
	int face_number;
	int CreateEachBaseMesh(int cloud_id);
	int CreateAllBaseMeshes();

	float InterPolateHeightField(float* heightField, float x, float y);

	int* isBaseBoundaryVertexList;
	void CreateBaseBoundaryVertexList();
	bool IsVertexOfTriangle(int id, int idx, int idy, int idz);
	int  CreateAllCloudMeshes();

private:
	void CreateCloudTopHeight_CTH_THICK();
	void ComputeGeoThick_CTH_THICK();
	void CreateCloudBottomHeight_CTH_THICK();

private:
	//draw
	//void PrintRunIfo(char* ifo);
	void ExportSatCloudIfo();
	void Export_PixelCloudIDList_CBH_CTH();

private:
	//noise 
	Perlin* perlin;

public:
	void Go(string input_path = ".", string output_path = "Cumulus_Output", string basemesh_folder = "BaseMesh", string cloudmesh_folder = "CloudMesh");
};
