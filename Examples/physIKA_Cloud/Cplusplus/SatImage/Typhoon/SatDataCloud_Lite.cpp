#pragma once
#include "SatDataCloud_Lite.h"
#include "WriteVTI.hpp"
#define SCALE (6656000 / 30);


SatDataCloud::SatDataCloud(void)
{
	ir1Data = NULL;
	ir2Data = NULL;
	ir3Data = NULL;
	ir4Data = NULL;
	visData = NULL;
	clcData = NULL;
	cttData = NULL;

	longitudeLatitudeTable = NULL;
	altitudeTable = NULL;

	ground_temperature_mat = NULL;

	satZenith_mat_E = NULL;
	satAzimuth_mat_E = NULL;
	satZenith_mat_F = NULL;
	satAzimuth_mat_F = NULL;
	sunZenithAzimuth_mat = NULL;

	irReflectanceData = NULL;
	vis_thick_data = NULL;
	ir4_thick_data = NULL;
	efficientRadius_data = NULL;

	pixelTypeList = NULL;
	cthList = NULL;

	thinCloudTList = NULL;

	extinctionPlane = NULL;

	for (int i = 0; i < 7; i++)
	{
		dataRange[i].x = MAXVAL;
		dataRange[i].y = -MAXVAL;
	}

	center_theta = 25.5 * M_PI / 180; // center_theta和center_pi是否需要变化？
	center_phi = 130 * M_PI / 180;

	perlin = new Perlin(4, 4, 1, 94);

	// SatDate初始化，权宜之计
	SatDate.year = -1;
	SatDate.month = -1;
	SatDate.day = -1;
}

SatDataCloud::~SatDataCloud(void)
{
	if (ir1Data != NULL)
		delete[] ir1Data;
	if (ir2Data != NULL)
		delete[] ir2Data;
	if (ir3Data != NULL)
		delete[] ir3Data;
	if (ir4Data != NULL)
		delete[] ir4Data;
	if (visData != NULL)
		delete[] visData;
	if (clcData != NULL)
		delete[] clcData;
	if (cttData != NULL)
		delete[] cttData;

	if (longitudeLatitudeTable != NULL)
		delete[] longitudeLatitudeTable;
	if (altitudeTable != NULL)
		delete[] altitudeTable;
	if (ground_temperature_mat != NULL)
		delete[] ground_temperature_mat;

	if (satZenith_mat_E != NULL)
		delete[] satZenith_mat_E;
	if (satAzimuth_mat_E != NULL)
		delete[] satAzimuth_mat_E;
	if (satZenith_mat_F != NULL)
		delete[] satZenith_mat_F;
	if (satAzimuth_mat_F != NULL)
		delete[] satAzimuth_mat_F;
	if (sunZenithAzimuth_mat != NULL)
		delete[] sunZenithAzimuth_mat;

	if (pixelTypeList != NULL)
		delete[] pixelTypeList;

	if (irReflectanceData != NULL)
		delete[] irReflectanceData;

	if (thinCloudTList != NULL)
		delete[] thinCloudTList;

	if (ir4_thick_data != NULL)
		delete[] ir4_thick_data;
	if (vis_thick_data != NULL)
		delete[] vis_thick_data;

	if (efficientRadius_data != NULL)
		delete[] efficientRadius_data;

	if (cthList != NULL)
		delete[] cthList;

	if (extinctionPlane != NULL)
		delete[] extinctionPlane;
}

void SatDataCloud::CreateLongLatTable(void) //No.4
{
	longitudeLatitudeTable = new float[WIDTH * HEIGHT * 2];

	float minlg = 100.57 * (M_PI / 180);
	float minlat = -10.38 * (M_PI / 180);
	float dlg = (169.41 - 100.57) * (M_PI / 180) / (WIDTH - 1);
	float temp = EARTH_RADIUS * cosf(22.5 * M_PI / 180);

	float Jd_leftbottom = sqrtf(pow(temp / cosf(minlat), 2) - temp * temp);
	if (minlat < 0)
		Jd_leftbottom *= -1;
	for (int i = 0; i < HEIGHT; i++)
	{
		for (int j = 0; j < WIDTH; j++)
		{
			longitudeLatitudeTable[2 * ((HEIGHT - 1 - i) * WIDTH + j) + 0] = (minlg + j * dlg);
			float Jd = Jd_leftbottom + GEO_INTERVAL * i;

			longitudeLatitudeTable[2 * ((HEIGHT - 1 - i) * WIDTH + j) + 1] = acosf(temp / (sqrt(temp * temp + Jd * Jd)));
			if (Jd < 0)
				longitudeLatitudeTable[2 * ((HEIGHT - 1 - i) * WIDTH + j) + 1] *= -1;
		}
	}
}

void SatDataCloud::CreateAltitudeTable() //No.20
{
	PrintRunIfo("CreateAltitudeTable");
	altitudeTable = new float[WIDTH * HEIGHT];
	for (int i = 0; i < WIDTH * HEIGHT; i++)
		altitudeTable[i] = 0;
}

void SatDataCloud::CreateSatZenithAzimuthIfo() //No.18
{
	PrintRunIfo("CreateSatZenithAzimuthIfo");
	satZenith_mat_E = new float[WIDTH * HEIGHT];
	satAzimuth_mat_E = new float[WIDTH * HEIGHT];

	satZenith_mat_F = new float[WIDTH * HEIGHT];
	satAzimuth_mat_F = new float[WIDTH * HEIGHT];

	//some constants
	const float SatelliteAltitude = 35860;			//km
	const float EarthRadius = EARTH_RADIUS;			//km
	float SatelliteLongitude = 123.05 * M_PI / 180; //FY2E
													// first compute the center angle formed by the arc between the intesection point (of  the satillite and the earth center)
													// and  the observation  point (longitude, latitude)
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float Longitude_dif = longitudeLatitudeTable[2 * (i * WIDTH + j) + 0] / 180 * M_PI - SatelliteLongitude;
			float CenterAngle = acos(cos(longitudeLatitudeTable[2 * (i * WIDTH + j) + 1]) * cos(Longitude_dif));
			satAzimuth_mat_E[i * WIDTH + j] = asin(sin(sin(Longitude_dif) / sin(CenterAngle)));
			float beta = atan((cos(CenterAngle) - EarthRadius / (SatelliteAltitude + EarthRadius)) / sin(CenterAngle));
			satZenith_mat_E[i * WIDTH + j] = M_PI / 2 - (CenterAngle + beta);
			//cout<<  satZenith_mat_E[i*WIDTH+j]<< " : "<<satAzimuth_mat_E[i*WIDTH+j]<<endl;
		}

	SatelliteLongitude = 112 * M_PI / 180; //FY2F
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float Longitude_dif = longitudeLatitudeTable[2 * (i * WIDTH + j) + 0] / 180 * M_PI - SatelliteLongitude;
			float CenterAngle = acos(cos(longitudeLatitudeTable[2 * (i * WIDTH + j) + 1]) * cos(Longitude_dif));
			satAzimuth_mat_F[i * WIDTH + j] = asin(sin(sin(Longitude_dif) / sin(CenterAngle)));
			float beta = atan((cos(CenterAngle) - EarthRadius / (SatelliteAltitude + EarthRadius)) / sin(CenterAngle));
			satZenith_mat_F[i * WIDTH + j] = M_PI / 2 - (CenterAngle + beta);
			//cout<<  satZenith_mat_F[i*WIDTH+j]<< " : "<<satAzimuth_mat_F[i*WIDTH+j]<<endl;
		}
}

int SatDataCloud::CreateSunZenithAzimuthIfo(Date date)     //No.19 == No.21
{
	extern vector<double> SunPosition(double year, double month, double day, double hour, double minute, double second,
		double height, double width, double offset, vector<double>& LongitudeLatitudeTable, vector<double>& AltitudeTable);
		
	sunZenithAzimuth_mat=new float[NFRAME*2*WIDTH*HEIGHT];

	vector<double> tempLongLatTable;
	for (int i = 0; i < 2 * WIDTH * HEIGHT; i++)
	{
		tempLongLatTable.push_back((double)longitudeLatitudeTable[i] * 180 / M_PI);
	}

	vector<double> tempAltitudeTable;
	for (int i = 0; i < WIDTH * HEIGHT; i++)
	{
		tempAltitudeTable.push_back(altitudeTable[i]);
	}

	// int year=date.year;
	// int month=date.month;
	// int day=date.day;
	int year = SatDate.year;
	int month = SatDate.month;
	int day = SatDate.day;	
	int second=0;
	int offset=0;
	int width=WIDTH;
	int height=HEIGHT;
	for(int nframe=0;nframe<NFRAME;nframe++)
	{
		int hour=(nframe/2);
		int minute=(nframe%2)*30;
		vector<double> tempSunZenithAzimuthTable = 
						SunPosition(year, month, day, hour, minute, second, height, width, offset, tempLongLatTable, tempAltitudeTable);
		for(int i=0;i<HEIGHT;i++)
		{
			for(int j=0;j<WIDTH;j++)
			{
					sunZenithAzimuth_mat[2*(nframe*WIDTH*HEIGHT+WIDTH*i+j)+0]=tempSunZenithAzimuthTable[2*(i*WIDTH+j)+0]*M_PI/180;
			        sunZenithAzimuth_mat[2*(nframe*WIDTH*HEIGHT+WIDTH*i+j)+1]=tempSunZenithAzimuthTable[2*(i*WIDTH+j)+1]*M_PI/180;
			}
		}
	}
	// FILE* fp=NULL;
	// fp = fopen("sunZenithAzimuth.dat", "wb");
	// fwrite(sunZenithAzimuth_mat,sizeof(float ),WIDTH*HEIGHT*2*NFRAME,fp);
	// fclose(fp);
	return 1;
}

void SatDataCloud::CreateSunZenithAzimuthIfoFromFile(const char* filename) //No.21
{
	PrintRunIfo("CreateSunZenithAzimuthIfo");
	sunZenithAzimuth_mat = new float[NFRAME * 2 * WIDTH * HEIGHT];

	FILE* fp = fopen(filename, "rb");
	fread(sunZenithAzimuth_mat, sizeof(float), WIDTH * HEIGHT * 2 * NFRAME, fp);
	for (int i = 0; i < NFRAME * 2 * WIDTH * HEIGHT; i += 2)
	{
		if (sunZenithAzimuth_mat[i] > 70 * M_PI / 180)
			sunZenithAzimuth_mat[i] = 70 * M_PI / 180;
		//cout<<"sun:  ("<<sunZenithAzimuth_mat[i]<<" , "<<sunZenithAzimuth_mat[i+1]<<" )"<<endl;
	}
	fclose(fp);
}

void SatDataCloud::ReadSatData(CString satStr, Date date, SatDataType channel) //No.5
{
	float* pData = new float[NFRAME * WIDTH * HEIGHT];
	for (int i = 0; i < NFRAME; i++)
	{

		int hour = i / 2;
		int minute = (i % 2) * 30;
		if(date.IsUsingHMS()==true)
		{
			hour = date.hour;
			minute = date.minute;
		}

		char str[256];
		TimeChannel2FileName(satStr, str, date, hour, minute, channel);
		ReadSingleSatData(str, pData + i * WIDTH * HEIGHT, channel, i);
	}
	for (int i = 0; i < NFRAME * WIDTH * HEIGHT; i++)
	{
		dataRange[(int)channel].x = min(dataRange[(int)channel].x, pData[i]);
		dataRange[(int)channel].y = max(dataRange[(int)channel].y, pData[i]);
	}

	switch (channel)
	{
	case IR1:
		ir1Data = pData;
		break;
	case IR2:
		ir2Data = pData;
		break;
	case IR3:
		ir3Data = pData;
		break;
	case IR4:
		ir4Data = pData;
		break;
	case VIS:
		visData = pData;
		break;
	case CLC:
		clcData = pData;
		break;
	case CTT:
		cttData = pData;
		break;
	}
}

void SatDataCloud::ReadSatData(const string& satStr, SatDataType channel)
{
	float* pData = new float[NFRAME * WIDTH * HEIGHT];
	for (int i = 0; i < NFRAME; i++)
	{
		ReadSingleSatData(const_cast<char*>(satStr.c_str()), pData + i * WIDTH * HEIGHT, channel, i);
	}
	for (int i = 0; i < NFRAME * WIDTH * HEIGHT; i++)
	{
		dataRange[(int)channel].x = min(dataRange[(int)channel].x, pData[i]);
		dataRange[(int)channel].y = max(dataRange[(int)channel].y, pData[i]);
	}

	switch (channel)
	{
	case IR1:
		ir1Data = pData;
		break;
	case IR2:
		ir2Data = pData;
		break;
	case IR3:
		ir3Data = pData;
		break;
	case IR4:
		ir4Data = pData;
		break;
	case VIS:
		visData = pData;
		break;
	case CLC:
		clcData = pData;
		break;
	case CTT:
		cttData = pData;
		break;
	}
}

void SatDataCloud::TimeChannel2FileName(CString satStr, char strFile[256], Date date, int hour, int minute, SatDataType channel) //No.11
{
	CString channelStr;
	switch (channel)
	{
	case IR1:
		channelStr = "_SEC_IR1_MLS_";
		break;
	case IR2:
		channelStr = "_SEC_IR2_MLS_";
		break;
	case IR3:
		channelStr = "_SEC_IR3_MLS_";
		break;
	case IR4:
		channelStr = "_SEC_IR4_MLS_";
		break;
	case VIS:
		channelStr = "_SEC_VIS_MLS_";
		break;
	case CLC:
		channelStr = "_CLC_MLT_OTG_";
		break;
	case CTT:
		channelStr = "_CTT_MLT_OTG_";
		break;
	}

	//FY2E_SEC_VIS_MLS_20120806_2200.AWX
	CString filename;
	filename = channelStr;
	// std::cout<<"filename : "<<filename<<endl;

	CString str;
	str.Format("%d", date.year);
	filename += str;
	// std::cout<<"filename : "<<filename<<endl;

	if (date.month < 10)
	{
		str.Format("0%d", date.month);
	}
	else
	{
		str.Format("%d", date.month);
	}
	filename += str;
	// std::cout<<"filename : "<<filename<<endl;

	if (date.day < 10)
	{
		str.Format("0%d_", date.day);
	}
	else
	{
		str.Format("%d_", date.day);
	}
	filename += str;
	// std::cout<<"filename : "<<filename<<endl;

	if (hour < 10)
	{
		str.Format("0%d", hour);
	}
	else
	{
		str.Format("%d", hour);
	}
	filename += str;
	// std::cout<<"filename : "<<filename<<endl;

	//switch (channel)
	//{
	//case IR1:
	//case IR2:
	//case IR3:
	//case IR4:
	//	satStr = "suli//SEC//";
	//	break;
	//case VIS:
	//	satStr = "suli//MLS//";
	//	break;
	//case CLC:
	//	satStr = "suli//CLC//";
	//	break;
	//case CTT:
	//	satStr = "suli//CTT//";
	//	break;
	//}
	if (minute < 30)
	{
		satStr += "FY2E";
		filename += "00.AWX";
	}
	else
	{
		satStr += "FY2F";
		filename += "30.AWX";
	}
	filename = satStr + filename;
	// std::cout<<"filename : "<<filename<<endl;
	//Cstring to char*
	_tcscpy(_T(strFile), filename.GetBuffer(filename.GetLength()));
}

bool SatDataCloud::ReadSingleSatData(char* filename, float* pData, SatDataType channel, int nframe) //No.6
{
	if (pData == NULL)
	{
		cout << "Please allocate memory for pData firstly!" << endl;
		return false;
		/*exit(1);*/
	}

	AwxFileFirstHeader firstHead;
	AwxFileGeoSatelliteSecondHeader secondHead;
	byte colorTable[256 * 3];
	unsigned int calibrationTable[1024];

	FILE* fp = NULL;
	fp = fopen(filename, "rb");
	if (fp == NULL)
	{
		cout << "Can not open the file:  " << filename << endl;
		return false;
	}

	fread(firstHead.strFileName, sizeof(char), 12, fp);
	fread(&(firstHead.iByteOrder), sizeof(short int), 1, fp);
	fread(&(firstHead.iFirstHeaderLength), sizeof(short int), 1, fp);
	fread(&(firstHead.iSecondHeaderLength), sizeof(short int), 1, fp);
	fread(&(firstHead.iFillSectionLength), sizeof(short int), 1, fp);
	fread(&(firstHead.iRecoderLength), sizeof(short int), 1, fp);
	fread(&(firstHead.iRecordsOfHeader), sizeof(short int), 1, fp);
	fread(&(firstHead.iRecordsOfData), sizeof(short int), 1, fp);
	fread(&(firstHead.iTypeOfProduct), sizeof(short int), 1, fp);
	fread(&(firstHead.iTypeOfCompress), sizeof(short int), 1, fp);
	fread(firstHead.strVersion, sizeof(char), 8, fp);
	fread(&(firstHead.iFlagOfQuality), sizeof(short int), 1, fp);

	fread(secondHead.strSatelliteName, sizeof(char), 8, fp);
	fread(&(secondHead.iYear), sizeof(short int), 1, fp);
	fread(&(secondHead.iMonth), sizeof(short int), 1, fp);
	fread(&(secondHead.iDay), sizeof(short int), 1, fp);
	fread(&(secondHead.iHour), sizeof(short int), 1, fp);
	fread(&(secondHead.iMinute), sizeof(short int), 1, fp);
	fread(&(secondHead.iChannel), sizeof(short int), 1, fp);
	fread(&(secondHead.iFlagOfProjection), sizeof(short int), 1, fp);
	fread(&(secondHead.iWidthOfImage), sizeof(short int), 1, fp);
	fread(&(secondHead.iHeightOfImage), sizeof(short int), 1, fp);
	fread(&(secondHead.iScanLineNumberOfImageTopLeft), sizeof(short int), 1, fp);
	fread(&(secondHead.iPixelNumberOfImageTopLeft), sizeof(short int), 1, fp);
	fread(&(secondHead.iSampleRatio), sizeof(short int), 1, fp);
	fread(&(secondHead.iLatitudeOfNorth), sizeof(short int), 1, fp);
	fread(&(secondHead.iLatitudeOfSouth), sizeof(short int), 1, fp);
	fread(&(secondHead.iLongitudeOfWest), sizeof(short int), 1, fp);
	fread(&(secondHead.iLongitudeOfEast), sizeof(short int), 1, fp);
	fread(&(secondHead.iCenterLatitudeOfProjection), sizeof(short int), 1, fp);
	fread(&(secondHead.iCenterLongitudeOfProjection), sizeof(short int), 1, fp);
	fread(&(secondHead.iStandardLatitude1), sizeof(short int), 1, fp);
	fread(&(secondHead.iStandardLatitude2), sizeof(short int), 1, fp);
	fread(&(secondHead.iHorizontalResolution), sizeof(short int), 1, fp);
	fread(&(secondHead.iVerticalResolution), sizeof(short int), 1, fp);
	fread(&(secondHead.iOverlapFlagGeoGrid), sizeof(short int), 1, fp);
	fread(&(secondHead.iOverlapValueGeoGrid), sizeof(short int), 1, fp);
	fread(&(secondHead.iDataLengthOfColorTable), sizeof(short int), 1, fp);
	fread(&(secondHead.iDataLengthOfCalibration), sizeof(short int), 1, fp);
	fread(&(secondHead.iDataLengthOfGeolocation), sizeof(short int), 1, fp);
	fread(&(secondHead.iReserved), sizeof(short int), 1, fp);

	if(SatDate.year == -1)
	{
		SatDate.year = secondHead.iYear;
		SatDate.month = secondHead.iMonth;
		SatDate.day = secondHead.iDay;
	}

	if (secondHead.iDataLengthOfColorTable != 0)
	{
		char temp[768];
		fread(temp, sizeof(char), 768, fp);

		for (int i = 0; i < 768; i++)
		{
			colorTable[i] = temp[i];
		}
	}

	if (secondHead.iDataLengthOfCalibration != 0)
	{
		short int temp[1024];
		fread(temp, sizeof(short int), 1024, fp);

		for (int i = 0; i < 1024; i++)
		{
			if (temp[i] < 0)
				calibrationTable[i] = temp[i] + 65536;
			else
				calibrationTable[i] = temp[i];
		}
	}
	//Read data
	int img_width = firstHead.iRecoderLength;
	int img_height = firstHead.iRecordsOfData;
	fseek(fp, img_width * firstHead.iRecordsOfHeader * sizeof(byte), SEEK_SET); //kkkkkkkkkkkkkkkkkkkkkkkkkkk  note here

	satelliteName = string(secondHead.strSatelliteName);

	//cout << string(secondHead.strSatelliteName) << endl;
	if (string(secondHead.strSatelliteName) != "GOES-16")
	{
		byte* dataTable = new byte[img_width * img_height];
		fread(dataTable, sizeof(byte), img_width * img_height, fp);

		float* img_data = new float[img_width * img_height];
		for (int i = 0; i < img_height; i++)
			for (int j = 0; j < img_width; j++)
			{
				if (channel == VIS || channel == IR1 || channel == IR2 || channel == IR3 || channel == IR4)
				{
					if (secondHead.iChannel != 4)
						img_data[i * img_width + j] = calibrationTable[4 * dataTable[i * img_width + j]] * 0.01; //IR1--4
					else
					{
						img_data[i * img_width + j] = min(1.0, calibrationTable[dataTable[i * img_width + j]] * 0.0001); //vis
					}
				}
				else
				{
					if (channel == CLC)
					{
						img_data[i * img_width + j] = dataTable[i * img_width + j];
					}
					if (channel == CTT)
					{
						img_data[i * img_width + j] = dataTable[i * img_width + j];
					}
				}
			}

		if (channel == VIS || channel == IR1 || channel == IR2 || channel == IR3 || channel == IR4)
		{
			if (WIDTH != img_width || HEIGHT != img_height)
			{
				IntepImgData(img_data, img_width, img_height, pData, WIDTH, HEIGHT);
			}
			else
			{
				for (int i = 0; i < img_height; i++)
					for (int j = 0; j < img_width; j++)
						pData[i * img_width + j] = img_data[i * img_width + j];
			}
		}

		if (channel == CLC || channel == CTT)
		{
			IntepImgData(nframe, channel, img_data, img_width, img_height, pData, WIDTH, HEIGHT);
		}
		delete[] img_data;
		delete[] dataTable;
	}
	else
	{
		fread(pData, sizeof(float), img_width * img_height, fp);
	}
	fclose(fp);
	return true;
}

void SatDataCloud::IntepImgData(float* img_data, int img_width, int img_height, float* pData, int width, int height) //No.43
{
	if (img_data == NULL || pData == NULL)
		return;

	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
		{
			float pox_y = float(i) / (height - 1) * (img_height - 1);
			int idy = int(pox_y);
			float delta_y = fabs(pox_y - idy);
			float pox_x = float(j) / (width - 1) * (img_width - 1);
			int idx = int(pox_x);
			float delta_x = fabs(pox_x - idx);

			if ((idx > 0) && (idx < img_width - 1) && (idy > 0) && (idy < img_height - 1))
			{
				float data1 = (1 - delta_x) * img_data[idy * img_width + idx] + delta_x * img_data[idy * img_width + idx + 1];
				float data2 = (1 - delta_x) * img_data[(idy + 1) * img_width + idx] + delta_x * img_data[(idy + 1) * img_width + idx + 1];
				pData[i * width + j] = (1 - delta_y) * data1 + delta_y * data2;
				continue;
			}
			if (idx == img_width - 1 && idy > 0 && idy < img_height - 1)
			{
				float data1 = (1 - delta_y) * img_data[idy * img_width + idx] + delta_y * img_data[(idy + 1) * img_width + idx];
				pData[i * width + j] = data1;
				continue;
			}
			if (idx == 0 && idy > 0 && idy < img_height - 1)
			{
				float data1 = (1 - delta_y) * img_data[idy * img_width + idx] + delta_y * img_data[(idy + 1) * img_width + idx];
				pData[i * width + j] = data1;
				continue;
			}
			if (idx > 0 && idx < img_width - 1 && idy == 0)
			{
				float data1 = (1 - delta_x) * img_data[idy * img_width + idx] + delta_x * img_data[(idy + 1) * img_width + idx + 1];
				pData[i * width + j] = data1;
				continue;
			}

			if (idx > 0 && idx < img_width - 1 && idy == img_height - 1)
			{
				float data1 = (1 - delta_x) * img_data[idy * img_width + idx] + delta_x * img_data[idy * img_width + idx + 1];
				pData[i * width + j] = data1;
				continue;
			}
			if (idx == 0 && idy == 0 || idx == 0 && idy == img_height - 1 || idx == img_width - 1 && idy == 0 || idx == img_width - 1 && idy == img_height - 1)
			{
				pData[i * width + j] = img_data[idy * img_width + idx];
			}
		}
}

//No.44
void SatDataCloud::IntepImgData(int nframe, SatDataType channel, float* img_data, int img_width, int img_height, float* pData, int width, int height)
{
	// FY2E-CLC:  Latitude: [ -45,55]   Longitude: [55,155]
	// FY2D-CLC:  Latitude: [ -60,60]   Longitude: [27,147]
	// FY2F-CTT:  Latitude: [ -45,55]    Longitude [62,162]
	if (img_data == NULL || pData == NULL)
		return;
	if (channel == CLC)
	{
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				float longitude = longitudeLatitudeTable[2 * (width * i + j) + 0] * 180 / M_PI;
				float latitude = longitudeLatitudeTable[2 * (width * i + j) + 1] * 180 / M_PI;

				if (nframe % 2 == 0) //FY2E
				{
					if (longitude > 55 && longitude < 155 && latitude > -45 && latitude < 55)
					{
						int img_idx = int((longitude - 55) / 0.1);
						int img_idy = int((55 - latitude) / 0.1);

						pData[width * i + j] = img_data[img_idy * img_width + img_idx];
					}
					else //out of  the region of FY2E
					{
						pData[width * i + j] = -1;
					}
				}
				else //FY2D
				{
					if (longitude > 27 && longitude < 147 && latitude > -60 && latitude < 60)
					{
						int img_idx = int((longitude - 27) / 0.1);
						int img_idy = int((60 - latitude) / 0.1);

						pData[width * i + j] = img_data[img_idy * img_width + img_idx];
					}
					else //out of  the region of FY2D
					{
						pData[width * i + j] = -1;
					}
				}
			}
	}
	if (channel == CTT)
	{
		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
			{
				float longitude = longitudeLatitudeTable[2 * (width * i + j) + 0] * 180 / M_PI;
				float latitude = longitudeLatitudeTable[2 * (width * i + j) + 1] * 180 / M_PI;

				if (nframe % 2 == 1) //FY2F
				{
					if (longitude > 62 && longitude < 162 && latitude > -45 && latitude < 55)
					{
						int img_idx = int((longitude - 62) / 0.05);
						int img_idy = int((55 - latitude) / 0.1);

						pData[width * i + j] = img_data[img_idy * img_width + img_idx];
					}
					else //out of  the region of FY2F
					{
						pData[width * i + j] = 0;
					}
				}
				else
				{
					pData[width * i + j] = 0;
				}
			}
	}
}

// Render.cpp使用部分
//void SatDataCloud::DrawSatData(SatDataType curChannel, int nframe) //No.7
//{
//	glDisable(GL_LIGHTING);
//
//	if (nframe >= NFRAME)
//	{
//		return;
//	}
//
//	float *pData;
//
//	switch (curChannel)
//	{
//	case IR1:
//		pData = ir1Data;
//		break;
//	case IR2:
//		pData = ir2Data;
//		break;
//	case IR3:
//		pData = ir3Data;
//		break;
//	case IR4:
//		pData = ir4Data;
//		break;
//	case VIS:
//		pData = visData;
//		break;
//	case CLC:
//		pData = clcData;
//		break;
//	case CTT:
//		pData = cttData;
//		break;
//	}
//
//	int width;
//	int height;
//	width = WIDTH;
//	height = HEIGHT;
//
//	if (pData == NULL)
//		return;
//
//	pData += nframe * width * height;
//
//	glBegin(GL_POINTS);
//	for (int i = 0; i < height; i++)
//		for (int j = 0; j < width; j++)
//		{
//			Vector3 color;
//			float grey;
//			if (curChannel == VIS)
//			{
//				grey = (pData[i * width + j] - dataRange[int(curChannel)].x) / (dataRange[int(curChannel)].y - dataRange[int(curChannel)].x);
//				color = Vector3(grey, grey, grey);
//			}
//			if (curChannel == IR1 || curChannel == IR2 || curChannel == IR3 || curChannel == IR4)
//			{
//				grey = 1.0 - (pData[i * width + j] - dataRange[int(curChannel)].x) / (dataRange[int(curChannel)].y - dataRange[int(curChannel)].x);
//				color = Vector3(grey, grey, grey);
//			}
//			if (curChannel == CLC)
//			{
//				switch (int(pData[i * WIDTH + j]))
//				{
//				case 0:
//					//color=Vector3(0,0.0,0.5);
//					//break;
//				case 1:
//					color = Vector3(0, 0.21, 0.40);
//					break;
//				case 11:
//				case 21:
//					color = Vector3(0.3, 0.3, 0.3);
//					break;
//				case 12:
//					color = Vector3(0.3, 0.3, 0.3);
//					break;
//				case 13:
//					//color=Vector3(1.0,0.7,0.7);
//					//break;
//				case 14:
//					//color=Vector3(1,1,1);
//					//break;
//				case 15:
//					color = Vector3(0.8, 0.8, 0.8);
//					break;
//				default:
//					color = Vector3(0.0, 0, 0);
//					break;
//				}
//			}
//			if (curChannel == CTT)
//			{
//				if (nframe % 2)
//				{
//					int cloudType = int(clcData[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
//					switch (cloudType)
//					{
//					case -1:
//						color = Vector3(0.8, 0.0, 0.0);
//						break;
//					case 0:
//						color = Vector3(0, 0.0, 0.8);
//						break;
//					case 1:
//						color = Vector3(0, 0.8, 0);
//						break;
//
//					case 11:
//					case 12:
//					case 13:
//					case 14:
//					case 15:
//					case 21:
//						float grey = pData[i * width + j] / 255.0; //(curData[i*WIDTH+j].x-dataRange[int(curChannel)].x)/(dataRange[int(curChannel)].y-dataRange[int(curChannel)].x);
//						color = Vector3(grey, grey, grey);
//						break;
//					}
//				}
//				else
//				{
//					color = Vector3(0.3, 0.3, 0.3);
//				}
//			}
//			glColor3fv(!color);
//			glVertex2f(float(j) / width, 1.0 - float(i) / height);
//		}
//	glEnd();
//}
//
//void SatDataCloud::DrawSimulationCube() //No.28
//{
//	glDisable(GL_LIGHTING);
//
//	glColor3f(1.0, 0, 0);
//	glBegin(GL_LINES);
//
//	glVertex3f(0, 0, 0);
//	glVertex3f(1, 0, 0);
//
//	glVertex3f(1, 0, 0);
//	glVertex3f(1, 1, 0);
//
//	glVertex3f(1, 1, 0);
//	glVertex3f(0, 1, 0);
//
//	glVertex3f(0, 1, 0);
//	glVertex3f(0, 0, 0);
//
//	glVertex3f(0, 0, 1);
//	glVertex3f(1, 0, 1);
//
//	glVertex3f(1, 0, 1);
//	glVertex3f(1, 1, 1);
//
//	glVertex3f(1, 1, 1);
//	glVertex3f(0, 1, 1);
//
//	glVertex3f(0, 1, 1);
//	glVertex3f(0, 0, 1);
//
//	glVertex3f(0, 0, 0);
//	glVertex3f(0, 0, 1);
//
//	glVertex3f(1, 0, 0);
//	glVertex3f(1, 0, 1);
//
//	glVertex3f(1, 1, 0);
//	glVertex3f(1, 1, 1);
//
//	glVertex3f(0, 1, 0);
//	glVertex3f(0, 1, 1);
//
//	glEnd();
//}
//
//void SatDataCloud::RenderFrame(SatDataType channel, int n_frame) //No.55
//{
//	//23:59GMT,  11/Dec/2012
//	CString channelStr;
//	if (n_frame % 2)
//		channelStr = "FY2F";
//	else
//		channelStr = "FY2E";
//	switch (channel)
//	{
//	case IR1:
//		channelStr += " IR1 ";
//		break;
//	case IR2:
//		channelStr += " IR2 ";
//		break;
//	case IR3:
//		channelStr += " WV ";
//		break;
//	case IR4:
//		channelStr += " MWIR ";
//		break;
//	case VIS:
//		channelStr += " VIS ";
//		break;
//	case CLC:
//		channelStr += " CLC ";
//		break;
//	case CTT:
//		channelStr += " CTT ";
//		break;
//	}
//	CString strFrameTime = channelStr;
//	CString str;
//	int hour = n_frame / 2;
//	str.Format(_T("%d"), hour);
//
//	strFrameTime += str;
//	strFrameTime += ":";
//	int minute = (n_frame % 2) * 30;
//	if (minute == 0)
//		str.Format(_T("%d0"), minute);
//	else
//		str.Format(_T("%d"), minute);
//	strFrameTime += str;
//
//	strFrameTime += "UTC 10/July/2013 ";
//	//Cstring to char*
//
//	char strFrame[100];
//	memset(strFrame, 0, 100);
//	_tcscpy(/*(wchar_t *)*/ strFrame, strFrameTime.GetBuffer(strFrameTime.GetLength()));
//	glRasterPos3f(0.3, 0.1, 0.1);
//	glColor3f(1.0, 0, 0);
//	for (int i = 0; i < 100; i++)
//		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, strFrame[i]);
//}
//
//void SatDataCloud::RenderFrame(DrawType dtype) //No.56
//{
//	CString channelStr;
//	switch (dtype)
//	{
//	case LongLat:
//		channelStr = "LongLat";
//		break;
//	case SunZA:
//		channelStr = "SunZA";
//		break;
//	case Seg:
//		channelStr = "Segmentation";
//		break;
//	case TopSurface:
//		channelStr = "Cloud Top Surface";
//		break;
//	case BottomSurface:
//		channelStr = "Cloud Bottom Surface";
//		break;
//	case Thick:
//		channelStr = "Thickness";
//		break;
//
//	case EffRadius:
//		channelStr = "Effective Radius";
//		break;
//
//	case Extinction:
//		channelStr = "Extinction";
//		break;
//
//	default:
//		channelStr = "Hello";
//		break;
//	}
//	char strFrame[100];
//	memset(strFrame, 0, 100);
//	_tcscpy(/*(wchar_t *)*/ strFrame, channelStr.GetBuffer(channelStr.GetLength()));
//	glRasterPos3f(0.3, 0.1, 0.1);
//	glColor3f(1.0, 0, 0);
//	for (int i = 0; i < 100; i++)
//		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, strFrame[i]);
//}

// Render.cpp -> Run()部分
void SatDataCloud::Run(Date date, string satStr, string savePath, string saveName, int height, int width) //No.41
{	
	this->HEIGHT = height;
	this->WIDTH = width;

	ReadSatData(satStr.c_str(), date, VIS);
	ReadSatData(satStr.c_str(), date, IR1);
	ReadSatData(satStr.c_str(), date, IR2);
	ReadSatData(satStr.c_str(), date, IR3);
	ReadSatData(satStr.c_str(), date, IR4);

	CreateLongLatTable();
	CreateAltitudeTable();
	CreateSatZenithAzimuthIfo();
	// CreateSunZenithAzimuthIfo(date); //CreateSunZenithAzimuthIfo(Date(2013,7,10));
	CreateSunZenithAzimuthIfo(Date(2013,7,10));   // 使用随便一个Date以适配参数，实际不使用
	//CreateSunZenithAzimuthIfoFromFile("sunZenithAzimuth.dat");

	//ReadSatData(Date(2013, 7, 10), VIS);
	//ReadSatData(Date(2013, 7, 10), IR1);
	//ReadSatData(Date(2013, 7, 10), IR2);
	//ReadSatData(Date(2013, 7, 10), IR3);
	//ReadSatData(Date(2013, 7, 10), IR4);
	//ReadSatData(Date(2013, 7, 10), CLC); // CLC和CTT是什么？
	//ReadSatData(Date(2013, 7, 10), CTT);

	CreateGroundTemperatureTable(satStr.c_str(), Date(2013, 6, 26), IR1);  //原文为Date(2013, 6, 26)，不知有何深意
	CreateGroundTemperatureTable(satStr.c_str(), Date(2013, 6, 26), IR2);
	CloudGroundSegment(); //云景分离
	//ModifyReflectance();
	IR4Temperature2Reflectance(); // 用于计算有效半径
	Classification();
	CreateCloudTopHeight();
	ComputeCloudProperties_MEA();
	ComputeGeoThick();
	//GenerateCloudParticlesFile(0,0,512,512,0, 1000);
	//GenerateCloudParticlesFile(savePath, saveName, 0, NFRAME);

	string tmp = savePath + saveName;
	GenerateVolumeFile(tmp, 0, NFRAME);

	//GenerateExtinctionFieldFile(0, NFRAME, 256);
	//GenerateCloudFiledFileEarth(0,NFRAME,512,512,128);
	//GenerateCloudFieldIfoFile(0,1);
	//GenerateIfoForDobashi();
}

void SatDataCloud::Run(const vector<string>& typName, const string& savePath, const string& saveName, int height, int width)
{
	this->HEIGHT = height;
	this->WIDTH = width;

	if(typName.size() != 5)
	{	
		cout<<"需要5个频段的数据"<<endl;
		return;
	}

	ReadSatData(typName[0], VIS);
	ReadSatData(typName[1], IR1);
	ReadSatData(typName[2], IR2);
	ReadSatData(typName[3], IR3);
	ReadSatData(typName[4], IR4);

	CreateLongLatTable();
	CreateAltitudeTable();
	CreateSatZenithAzimuthIfo();
	CreateSunZenithAzimuthIfo(Date(2013,7,10));   // 使用随便一个Date以适配参数，实际不使用

	CreateGroundTemperatureTable(IR1, 0);
	CreateGroundTemperatureTable(IR2, 0);

	CloudGroundSegment(); //云景分离
	IR4Temperature2Reflectance(); // 用于计算有效半径
	Classification();
	CreateCloudTopHeight();
	ComputeCloudProperties_MEA();
	ComputeGeoThick();

	string tmp = savePath + saveName;
	GenerateVolumeFile(tmp, 0, NFRAME);
}


void SatDataCloud::GenerateExtinctionFieldFile(int startFrame, int endFrame, int Z_Res) //No.2
{
	float maxThick = -MAXVAL;
	float minThick = MAXVAL;

	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] > 0)
				{
					float curThick = geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					maxThick = max(maxThick, curThick);
					minThick = min(minThick, curThick);
				}
			}
	}
	//float scale =500;
	//float scale =300;
	//float scale =150; better
	float scale = 1;
	float geo_delta = 0;

	float scale_thin_cirrus = 1;
	float geo_delta_cirrus = 0;

	float* tempExtinctionField = new float[WIDTH * HEIGHT * Z_Res];
	memset(tempExtinctionField, 0, WIDTH * HEIGHT * Z_Res * sizeof(float));
	float maxHeight = -MAXVAL;
	float minHeight = MAXVAL;
	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				float cth = cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j];
				//if(geo_thick_data[nframe*WIDTH*HEIGHT+WIDTH*i+j]>500)
				//{
				//	cout<<"THICK !!HERE"<<endl;
				//}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 1 || pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 3)
				{
					float cbh_scale = cth - geo_delta - scale * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					maxHeight = max(maxHeight, cth);
					minHeight = min(minHeight, cbh_scale);
				}

				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 2)
				{
					float cbh_scale = cth - geo_delta_cirrus - scale_thin_cirrus * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					maxHeight = max(maxHeight, cth);
					minHeight = min(minHeight, cbh_scale);
				}
			}
	}
	float dZ = (maxHeight - minHeight) / (Z_Res - 1);
	cout << "Sampling: Height:  (" << minHeight << " , " << maxHeight << ")   Z interval: " << dZ << endl;

	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 0)
				{
					for (int z = 0; z < Z_Res; z++)
					{
						tempExtinctionField[z * WIDTH * HEIGHT + WIDTH * i + j] = 0.0;
					}
				}
				else
				{
					float cth = cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					float cbh_scale;

					if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 1 || pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 3)
					{
						cbh_scale = cth - geo_delta - scale * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					}
					if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] == 2)
					{
						cbh_scale = cth - geo_delta_cirrus - scale_thin_cirrus * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					}

					int incre = 1;
					int top_Z = int((cth - minHeight) / dZ) + 1;
					int bottom_Z = int((cbh_scale - minHeight) / dZ);
					if (top_Z > Z_Res - 1)
						top_Z = Z_Res - 1;
					if (bottom_Z < 0)
						bottom_Z = 0;

					for (int z = 0; z < Z_Res; z++)
					{
						if (z >= bottom_Z && z <= top_Z)
						{
							float temp = extinctionPlane[WIDTH * i + j];
							tempExtinctionField[z * WIDTH * HEIGHT + WIDTH * i + j] = temp;
						}
						else
						{
							tempExtinctionField[z * WIDTH * HEIGHT + WIDTH * i + j] = 0.0;
						}
					}
				} //else
			}
		FILE* fp = NULL;
		char fileName[100];

		for (int z = 0; z < Z_Res; z++)
			for (int i = 0; i < HEIGHT; i++)
				for (int j = i + 1; j < WIDTH; j++)
				{
					float tempExt = tempExtinctionField[z * WIDTH * HEIGHT + i * WIDTH + j];
					tempExtinctionField[z * WIDTH * HEIGHT + i * WIDTH + j] = tempExtinctionField[z * WIDTH * HEIGHT + j * HEIGHT + i];
					tempExtinctionField[z * WIDTH * HEIGHT + j * HEIGHT + i] = tempExt;
				}
		sprintf(fileName, "satclouds\\satcloud%d.dat", nframe);
		fp = fopen(fileName, "wb");
		if (fp != NULL)
		{
			fwrite(tempExtinctionField, sizeof(float), WIDTH * HEIGHT * Z_Res, fp);
			fclose(fp);
			fp = NULL;
		}
		else
		{
			cout << "can not open file for recording extinction field" << endl;
			return;
		}
		cout << "satclouds data: " << nframe << endl;
	}
}

void SatDataCloud::Classification() //No.29      //cloud-ground labeling
{
	//  pixelTypeList  0-ground, 1-cloud, 2-thin cloud(thin cirrus)
	PrintRunIfo("Classification");
	if (pixelTypeList == NULL)
	{
		cout << "Please do cloud-ground segmentation first!" << endl;
		exit(1);
	}
	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		SegmentSatDataIRWV(nframe);
		LabelCirrus(nframe);
		//SegmentSatDataKMeans(nframe);
	}
}

void SatDataCloud::LabelCirrus(int nframe) //No.30
{
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				float irT = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
				float irWV = ir3Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];

				float temp = ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j] - ground_temperature_mat_ir2[nframe * WIDTH * HEIGHT + i * WIDTH + j] + 1.6;

				//if (ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] < 273 && ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] - ir2Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] > temp)
				if (ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] < 273 && ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] - ir2Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 1.4)
					pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 2; //thin cloud  or thin cirrus

				//if(irWV-irT>0)
				//	pixelTypeList[nframe*WIDTH*HEIGHT+i*WIDTH+j]=3;
			}
		}
}

void SatDataCloud::SegmentSatDataIRWV(int nframe) //No.53
{
	float* pData_IR = new float[WIDTH * HEIGHT];
	float* pData_WV = new float[WIDTH * HEIGHT];
	char str[256];
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 1)
			{
				float avg_IR = 0.0;
				float avg_WV = 0.0;
				int negborCount = 0.0;
				for (int k = -1; k <= 1; k++)
					for (int l = -1; l <= 1; l++)
					{
						int idy = i + k;
						int idx = j + l;
						if (idy >= 0 && idy < HEIGHT && idx >= 0 && idx < WIDTH && pixelTypeList[nframe * WIDTH * HEIGHT + idy * WIDTH + idx] > 0)
						{
							avg_IR += ir1Data[nframe * WIDTH * HEIGHT + idy * WIDTH + idx];
							avg_WV += ir3Data[nframe * WIDTH * HEIGHT + idy * WIDTH + idx];
							negborCount++;
						}
					}
				avg_IR /= negborCount;
				avg_WV /= negborCount;
				pData_IR[i * WIDTH + j] = avg_IR;
				pData_WV[i * WIDTH + j] = avg_WV;
			}
		}
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 1)
			{
				float slope;
				bool isOne = FindSlope(nframe, i, j, slope, pData_IR, pData_WV);
				if (isOne)
				{
					float irT = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float irWV = ir3Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];

					if (slope < 0.1)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1;
					if (slope >= 0.1 && slope < 0.3)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1;
					if (slope >= 0.3 && slope < 0.5)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 3;
					if (slope >= 0.5 && slope < 0.65)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 3;
					if (slope >= 0.65)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 3;
					//if(visData[nframe*512*512+i*512+j]<0.15)
					// 	pixelTypeList[nframe*WIDTH*HEIGHT+i*WIDTH+j]=1;

					float tempH = (ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j] - ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j]) * 1000 / 6.48 + altitudeTable[WIDTH * i + j];
					if (tempH < 2500)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1;
				}
				else
				{
					pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1;
				}
			}
		}
	delete[] pData_IR;
	delete[] pData_WV;
}

bool SatDataCloud::FindSlope(int nframe, int i, int j, float& slope, float* pdata_ir, float* pdata_wv) //No.54
{
	vector<int> neigborList;
	neigborList.clear();

	int searchExt = 3;
	while (neigborList.size() < 24)
	{
		neigborList.clear();
		int iIndex = i;
		int jIndex = j;
		int iStart = max(i - searchExt, 0);
		int iEnd = min(i + searchExt, HEIGHT);

		int jStart = max(j - searchExt, 0);
		int jEnd = min(j + searchExt, WIDTH);

		for (int ii = iStart; ii < iEnd; ii++)
			for (int jj = jStart; jj < jEnd; jj++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + ii * WIDTH + jj] > 0)
				{
					neigborList.push_back(ii);
					neigborList.push_back(jj);
				}
			}
		searchExt += 2;
		if (searchExt > WIDTH)
		{
			return false;
		}
	}

	//fit a with least square
	float A11 = 0.0;
	float A12 = 0.0;
	float A21 = 0.0;
	float A22 = 0.0;
	float B1 = 0.0;
	float B2 = 0.0;
	for (int n = 0; n < neigborList.size() / 2; n++)
	{
		int ii = neigborList[2 * n + 0];
		int jj = neigborList[2 * n + 1];
		float yn = pdata_wv[ii * WIDTH + jj];
		float xn = pdata_ir[ii * WIDTH + jj];
		A11 += xn * xn;
		A12 += xn;
		B1 += xn * yn;
		B2 += yn;
	}
	A21 = A12;
	A22 = neigborList.size() / 2;

	float Det = A11 * A22 - A12 * A21;
	if (fabs(Det) < F_ZERO)
	{
		return false;
	}
	float a, b;
	a = (A22 * B1 - A12 * B2) / Det;
	b = (-A21 * B1 + A11 * B2) / Det;

	slope = a;
	return true;
}

void SatDataCloud::CloudGroundSegment() //No.13
{
	PrintRunIfo("CLoud-Ground");
	pixelTypeList = new int[NFRAME * WIDTH * HEIGHT];

	int count = 0;
	float* pData = new float[WIDTH * HEIGHT];
	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		for (int i = 0; i < HEIGHT - 1; i++)
			for (int j = 0; j < WIDTH - 1; j++)
				pData[i * WIDTH + j] = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];

		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 0; //ground

				if (satelliteName != "GOES-16")
				{
					float tmp = ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					if (tmp - pData[i * WIDTH + j] > CLOUD_GROUND_SEG_THRESHOLD)
						pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1; //cloud
				}
				if (ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] - ir2Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 1 && ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] < 233)
					pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1; //thin cloud,partial cloud

				//if (sunZenithAzimuth_mat[2 * (i * WIDTH + j) + 0] < 60.0 * M_PI / 180 && visData[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0.5)
				//	pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1; //clouds have large reflectance
				if(visData[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0.45)
					pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 1; //clouds have large reflectance

				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
					count++;
			}
	}
	cout << "has cloud nums :" << count << endl;
	delete[] pData;
}

void SatDataCloud::IR4Temperature2Reflectance() //No.22
{
	irReflectanceData = new float[NFRAME * WIDTH * HEIGHT];
	float F0[4] = { 0.17904e6, 0.1152e6, 0.9862e6, 11.02e6 };

	float minR = 9999;
	;
	float maxR = -9999;
	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
				{
					float T4 = ir4Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float T1 = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float T2 = ir2Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float L4 = PlanckFunction(3, T4);
					float L1 = PlanckFunction(3, T1);

					float sunZenith = sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + i * WIDTH + j) + 0];
					float theta0 = (2 * M_PI * 215 - 1) / 365;
					//cout<<sunZenith<<endl;
					float avgDis_Sun_Earth = 1.000110 + 0.034221 * cos(theta0) + 0.001280 * sin(theta0) + 0.000719 * cos(2 * theta0) + 0.000077 * sin(2 * theta0);
					float R = F0[3] / M_PI * avgDis_Sun_Earth * fabs(cos(sunZenith));
					float Re = (L4 - L1) / (R - L1);

					irReflectanceData[nframe * WIDTH * HEIGHT + i * WIDTH + j] = Re;
				}
			}
	}
}

float SatDataCloud::PlanckFunction(int channel, float T) //No.31
{
	float c = 3.0e8;
	float h = 6.626e-34;
	float k = 1.3806e-23;
	float w[4] = { 10.8e-6, 12.0e-6, 6.95e-6, 3.75e-6 };
	float L = 2 * h * c * c / (pow(w[channel], 5) * (exp(h * c / (w[channel] * k * T)) - 1));
	return L;
}

void SatDataCloud::CreateGroundTemperatureTable(CString satStr, Date startDate, SatDataType channel) //No.9
{
	float* ground_temperature_mat_temp = new float[NFRAME * WIDTH * HEIGHT];

	float* pData = new float[WIDTH * HEIGHT];

	for (int nframe = 0; nframe < NFRAME; nframe++)
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				ground_temperature_mat_temp[nframe * WIDTH * HEIGHT + i * WIDTH + j] = -9999;
			}
	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		for (int day = 0; day < 15; day++)
		{
			//Attention: A more accurate version should be given.
			// Date date = startDate;
			Date date = SatDate;
			date.day += day;
			if (date.day > 30)
			{
				date.month++;
				date.day -= 30;
			}
			int hour = (nframe / 2);
			int minute = (nframe % 2) * 30;

			bool readOk = ReadFixedTimeAwxData(satStr, pData, date, hour, minute, channel, nframe);

			for (int i = 0; i < HEIGHT; i++)
				for (int j = 0; j < WIDTH; j++)
				{
					if (!readOk)
						pData[i * WIDTH + j] = -MAXVAL;
					if (ground_temperature_mat_temp[nframe * WIDTH * HEIGHT + i * WIDTH + j] < pData[i * WIDTH + j])
						ground_temperature_mat_temp[nframe * WIDTH * HEIGHT + i * WIDTH + j] = pData[i * WIDTH + j];
				}
		}
		cout << "Ground Tempeature:  " << nframe << endl;
	}
	if (channel == IR1)
	{
		std::ofstream outfile("./ground_temp_IR1.dat", std::ios_base::binary | std::ios_base::out);
		unsigned int gSize = (unsigned int)(WIDTH * HEIGHT);
		outfile.write((char *)&gSize, sizeof(gSize));
		outfile.write((char *)ground_temperature_mat_temp, sizeof(float) * gSize);

		//for (int i = 0; i < gSize; i++)
		//	cout << ground_temperature_mat_temp[i] << endl;

		ground_temperature_mat = ground_temperature_mat_temp;
	}
	else if(channel == IR2)
	{
		std::ofstream outfile("./ground_temp_IR2.dat", std::ios_base::binary | std::ios_base::out);
		unsigned int gSize = (unsigned int)(WIDTH * HEIGHT);
		outfile.write((char *)&gSize, sizeof(gSize));
		outfile.write((char *)ground_temperature_mat_temp, sizeof(float) * gSize);

		ground_temperature_mat_ir2 = ground_temperature_mat_temp;
	}

	delete[] pData;
}

void SatDataCloud::CreateGroundTemperatureTable(SatDataType channel) //No.9
{
    if (channel == IR1)
	{
		float* ground_temperature_mat_temp = new float[WIDTH * HEIGHT];
		ground_temperature_mat = new float[WIDTH * HEIGHT];

		std::ifstream infile("./ground_temp_IR1.dat", ifstream::binary);
		unsigned int gSize;
		infile.read((char *)(&gSize), sizeof(gSize));
		infile.read((char *)ground_temperature_mat_temp, sizeof(float) * gSize);

		IntepImgData(ground_temperature_mat_temp, WIDTH, HEIGHT, ground_temperature_mat, WIDTH, HEIGHT);

        delete[] ground_temperature_mat_temp;
	}
	else if (channel == IR2)
	{
		float* ground_temperature_mat_ir2_temp = new float[WIDTH * HEIGHT];
		ground_temperature_mat_ir2 = new float[WIDTH * HEIGHT];

		std::ifstream infile("./ground_temp_IR2.dat", ifstream::binary);
		unsigned int gSize;
		infile.read((char *)(&gSize), sizeof(gSize));
		infile.read((char *)ground_temperature_mat_ir2_temp, sizeof(float) * gSize);

		IntepImgData(ground_temperature_mat_ir2_temp, WIDTH, HEIGHT, ground_temperature_mat_ir2, WIDTH, HEIGHT);

        delete[] ground_temperature_mat_ir2_temp;
	}
}

void SatDataCloud::CreateGroundTemperatureTable(SatDataType channel, int difference)
{
	float* temp = nullptr;
	if (channel == IR1)
	{
		temp = ir1Data;
	}
	else if (channel == IR2)
	{
		temp = ir2Data;
	}

	float* temperatureTable = new float[NFRAME * WIDTH * HEIGHT];
	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		float max = -9999;
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				if(temp[nframe * WIDTH * HEIGHT + i * WIDTH + j] > max)
					max = temp[nframe * WIDTH * HEIGHT + i * WIDTH + j];
			}
		}

		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				temperatureTable[nframe * WIDTH * HEIGHT + i * WIDTH + j] = max;
			}
		}
	}

	if (channel == IR1)
	{
		ground_temperature_mat = temperatureTable;
	}
	else if (channel == IR2)
	{
		ground_temperature_mat_ir2 = temperatureTable;
	}
}

bool SatDataCloud::ReadFixedTimeAwxData(CString satStr, float* pData, Date date, int hour, int minute, SatDataType channel, int nframe) //No.8
{
	if (pData == NULL)
	{
		cout << "pData is NULL" << endl;
		return false;
	}
	char str[256];
	TimeChannel2FileName(satStr,str, date, hour, minute, channel);
	return ReadSingleSatData(str, pData, channel, nframe);
}

void SatDataCloud::CreateCloudTopHeight() //No.15
{
	PrintRunIfo("CTH");
	if (ir1Data == NULL || pixelTypeList == NULL || ground_temperature_mat == NULL)
		return;

	thinCloudTList = new float[NFRAME * WIDTH * HEIGHT];
	cthList = new float[NFRAME * WIDTH * HEIGHT];
	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 0)
				{
					cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 0;
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 1 || pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 3)
				{
					cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = (ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j] - ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j]) * 1000 / 6.48 + altitudeTable[WIDTH * i + j];
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 2)
				{
					float T_efficient;
					float a, b;
					float temp = ir3Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float T_right = ir3Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float T_left = ir3Data[nframe * WIDTH * HEIGHT + i * WIDTH + j] - 30;
					if (FindaAndb(nframe, i, j, a, b))
					{
						while ((T_right - T_left) > 5)
						{
							float f_left = a * PlanckFunction(0, T_left) + b - PlanckFunction(2, T_left);
							float f_right = a * PlanckFunction(0, T_right) + b - PlanckFunction(2, T_right);
							if (f_left * f_right > 0)
							{
								pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 3;
								T_left = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
								break;
							}
							float T_middle = (T_right + T_left) / 2;
							float f_middle = a * PlanckFunction(0, T_middle) + b - PlanckFunction(2, T_middle);

							if (f_left * f_middle < 0)
							{
								T_right = T_middle;
							}

							if (f_right * f_middle < 0)
							{
								T_left = T_middle;
							}
						} //while
						T_efficient = T_left;
						thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = T_efficient;
						cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = (ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j] - T_left) * 1000 / 6.48 + altitudeTable[WIDTH * i + j];
					} //if find
					else
					{
						thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = T_left;
						cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = (ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j] - thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j]) * 1000 / 6.48 + altitudeTable[WIDTH * i + j];

						cout << "Find a and b failure!" << endl;
					}
				} //if ==2
				cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] = max((float)50, cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j]);          //50?
			} //for for
		cout << "CTH: " << nframe << endl;
	}
}

bool SatDataCloud::FindaAndb(int nframe, int i, int j, float& a, float& b) //No.14
{
	vector<int> neigborList;
	neigborList.clear();

	int searchExt = 1;
	while (neigborList.size() < 8)
	{
		neigborList.clear();
		int iIndex = i;
		int jIndex = j;
		int iStart = max(i - searchExt, 0);
		int iEnd = min(i + searchExt, HEIGHT);

		int jStart = max(j - searchExt, 0);
		int jEnd = min(j + searchExt, WIDTH);

		for (int ii = iStart; ii < iEnd; ii++)
			for (int jj = jStart; jj < jEnd; jj++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + ii * WIDTH + jj] == 2)
				{
					neigborList.push_back(ii);
					neigborList.push_back(jj);
				}
			}
		searchExt += 2;
	}
	//fit a with least square
	float A11 = 0.0;
	float A12 = 0.0;
	float A21 = 0.0;
	float A22 = 0.0;
	float B1 = 0.0;
	float B2 = 0.0;
	for (int n = 0; n < neigborList.size() / 2; n++)
	{
		int ii = neigborList[2 * n + 0];
		int jj = neigborList[2 * n + 1];
		float xn = PlanckFunction(2, ir3Data[nframe * WIDTH * HEIGHT + ii * WIDTH + jj]);
		float yn = PlanckFunction(0, ir1Data[nframe * WIDTH * HEIGHT + ii * WIDTH + jj]);
		A11 += yn * yn;
		A12 += yn;
		B1 += xn * yn;
		B2 += xn;
	}
	A21 = A12;
	A22 = neigborList.size() / 2;

	float Det = A11 * A22 - A12 * A21;
	if (fabs(Det) < F_ZERO)
	{
		return false;
	}
	a = (A22 * B1 - A12 * B2) / Det;
	b = (-A21 * B1 + A11 * B2) / Det;

	return true;
}

void SatDataCloud::ComputeCloudProperties_MEA() //No.23
{
	PrintRunIfo("Cloud Properites");
	//refIdx1 : Imaginary part of the refractive index for liquid water wavelength1
	//refIdx2: Imaginary part of the refractive index for liquid water wavelength2
	//A1: Ground surface albedo in wavelength1
	//A2: Ground surface albedo in wavelength2

	float wavelength1 = 0.7;
	float wavelength2 = 3.75;
	float refIdx1 = 1.64e-8;
	float refIdx2 = 4.0e-4;
	float A1 = 0.1;
	float A2 = 0.2;

	vis_thick_data = new float[NFRAME * WIDTH * HEIGHT];
	ir4_thick_data = new float[NFRAME * WIDTH * HEIGHT];
	efficientRadius_data = new float[NFRAME * WIDTH * HEIGHT];

	struct ReflectanceSampleIfo
	{
		float refValue;
		float thick1;
		float thick2;
		float radius;
	};

	int sampleRes = 50;
	ReflectanceSampleIfo* reflectanceSampleList = new ReflectanceSampleIfo[sampleRes];

	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		float* satZenith_mat;
		float* satAzimuth_mat;
		if (nframe % 2)
		{
			satZenith_mat = satZenith_mat_F;   // ?
			satAzimuth_mat = satAzimuth_mat_F;
		}
		else
		{
			satZenith_mat = satZenith_mat_E;
  			satAzimuth_mat = satAzimuth_mat_E;
		}

		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 1)
				{
					// a vector of the effective radius  from 3 ��m to 30 ��m
					int MaxRadius = 30;
					int MinRadius = 3;
					float delta_radius = (MaxRadius - MinRadius) / float(sampleRes - 1);

					float u = fabs(cos(satZenith_mat[WIDTH * i + j]));
					float u0 = cos(sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + i * WIDTH + j) + 0]);
					float phi = satAzimuth_mat[HEIGHT * i + j] - sunZenithAzimuth_mat[2 * (nframe * WIDTH * WIDTH + i * WIDTH + j) + 1];
					//float k1=2*M_PI/wavelength1;
					//float k2=2*M_PI/wavelength2;
					//float kk1=4*M_PI*refIdx1/wavelength1;
					//float kk2=4*M_PI*refIdx2/wavelength2;
					//escape function
					float K0u = 3.0 / 7 * (1 + 2 * u);
					float K0u0 = 3.0 / 7 * (1 + 2 * u0);

					for (int sample = 0; sample < sampleRes; sample++)
					{
						float curEfficientRadius = MinRadius + sample * delta_radius;
						//asymmetry parameter
						float g1 = 0.809 + 3.387 * 0.001 * curEfficientRadius; //1-(0.12+0.5*powf((k1*curEfficientRadius),-2.0/3)-0.15*kk1*curEfficientRadius);
						float g2 = 0.726 + 6.652 * 0.001 * curEfficientRadius; //1-(0.12+0.5*powf((k2*curEfficientRadius),-2.0/3)-0.15*kk2*curEfficientRadius);

						//single scattering albedo
						//	float ka=5*M_PI*refIdx2*(1-kk2*curEfficientRadius)/wavelength2*(1+0.34*(1-expf(-8*wavelength2/curEfficientRadius)));
						//   float ke=1.5/curEfficientRadius*(1+1.1/powf((k2*curEfficientRadius),2.0/3));
						float w02 = 1 - 0.025 - 0.0122 * curEfficientRadius; //1-ka/ke;
						float w01 = 1.0;									 //no absorbing for vis channel

						//Calculate scattering angle ��
						float theta = acos(-u * u0 + sin(sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + i * WIDTH + j) + 0]) * sin(satZenith_mat[WIDTH * i + j]) * cos(phi));

						//Calculate Henyey-Greenstein phase function from g1
						float phase1 = (1 - g1 * g1) / powf((1 + g1 * g1 - 2 * g1 * cos(theta)), 1.5);

						float A = 3.944;
						float B = -2.5;
						float C = 10.664;
						float RInf01 = (A + B * (u + u0) + C * u * u0 + phase1) / (4 * (u + u0));
						float visR = visData[nframe * HEIGHT * WIDTH + i * HEIGHT + j];
						//float t1=1.0/(K0u*K0u0/(RInf01-visData[nframe*512*512+i*512+j])-A1/(1-A1));
						float t1 = (RInf01 - visData[nframe * HEIGHT * WIDTH + i * HEIGHT + j]) / (K0u * K0u0);
						float alpha = 1.07;
						float Thickness1 = (1.0 / t1 - alpha) / (0.75 * (1 - g1));

						if (Thickness1 < 0 || Thickness1 > MAXVAL)
						{
							/*		   cout<<"water thickness too small or too greater:"<<endl;*/
							Thickness1 = 0.5;
						}

						float xi1 = 2 * M_PI * curEfficientRadius / wavelength1;
						float xi2 = 2 * M_PI * curEfficientRadius / wavelength2;
						float Thickness2 = Thickness1 * powf(wavelength2 / wavelength1, 2.0 / 3) * (1.1 + powf(xi2, 2.0 / 3)) / (1.1 + powf(xi1, 2.0 / 3));

						float x2 = sqrtf(3 * (1 - g2) * (1 - w02)) * Thickness2;
						float y2 = 4 * sqrtf((1 - w02) / (3 * (1 - g2)));
						float tc = sinh(y2) / sinh(alpha * y2 + x2);
						float t2 = tc - (4.86 - 13.08 * u * u0 + 12.76 * u * u * u0 * u0) * expf(x2) / powf(Thickness2, 3);
						float a2 = expf(-y2) - tc * expf(-x2 - y2);

						float phase2 = (1 - g2 * g2) / powf((1 + g2 * g2 - 2 * g2 * cos(theta)), 1.5);
						float RInf02 = (A + B * (u + u0) + C * u * u0 + phase2) / (4 * (u + u0));

						float mu = K0u0 * K0u / RInf02;
						mu = mu * (1 - 0.05 * y2);
						float curReflectance = RInf02 * expf(-y2 * mu) - (expf(-x2 - y2) - t2 * A2 / (1 - A2 * a2)) * t2 * K0u * K0u0;

						reflectanceSampleList[sample].refValue = curReflectance;
						reflectanceSampleList[sample].thick1 = Thickness1;
						reflectanceSampleList[sample].thick2 = Thickness2;
					}

					float minDif = 9999;
					int minSample = 0;
					for (int sample = 0; sample < sampleRes; sample++)
					{
						float visR = visData[nframe * WIDTH * HEIGHT + i * WIDTH + j];
						float irR = irReflectanceData[nframe * WIDTH * HEIGHT + i * WIDTH + j];
						float sampleIrR = reflectanceSampleList[sample].refValue;
						float dif = fabs(irR - sampleIrR);
						if (dif < minDif)
						{
							minDif = dif;
							minSample = sample;
						}
					}
					vis_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = reflectanceSampleList[minSample].thick1;
					ir4_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = reflectanceSampleList[minSample].thick2;
					reflectanceSampleList[minSample].radius = MinRadius + minSample * delta_radius;
					efficientRadius_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = reflectanceSampleList[minSample].radius;
					// cout<<"Raidus: "<< reflectanceSampleList[minSample].radius<<endl;
				}

				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 3 || pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 2)
				{
					// a vector of the effective radius  from 3 ��m to 30 ��m
					int MaxRadius = 150;
					int MinRadius = 30;
					float delta_radius = (MaxRadius - MinRadius) / float(sampleRes - 1);

					float u = fabs(cos(satZenith_mat[WIDTH * i + j]));
					float u0 = cos(sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + i * WIDTH + j) + 0]);
					float phi = satAzimuth_mat[HEIGHT * i + j] - sunZenithAzimuth_mat[2 * (nframe * WIDTH * WIDTH + i * WIDTH + j) + 1];
					/*			float k1=2*M_PI/wavelength1;
					float k2=2*M_PI/wavelength2;
					float kk1=4*M_PI*refIdx1/wavelength1;
					float kk2=4*M_PI*refIdx2/wavelength2;*/

					//escape function
					float K0u = 3.0 / 7 * (1 + 2 * u);
					float K0u0 = 3.0 / 7 * (1 + 2 * u0);

					for (int sample = 0; sample < sampleRes; sample++)
					{
						float curEfficientRadius = MinRadius + sample * delta_radius;

						//asymmetry parameter
						float g1 = 0.74; //1-(0.12+0.5*powf((k1*curEfficientRadius),-2.0/3)-0.15*kk1*curEfficientRadius);
						float g2 = 0.74; //1-(0.12+0.5*powf((k2*curEfficientRadius),-2.0/3)-0.15*kk2*curEfficientRadius);

						//single scattering albedo
						//		float ka=5*M_PI*refIdx2*(1-kk2*curEfficientRadius)/wavelength2*(1+0.34*(1-expf(-8*wavelength2/curEfficientRadius)));
						//	    float ke=1.5/curEfficientRadius*(1+1.1/powf((k2*curEfficientRadius),2.0/3));

						float l = 1.8 * curEfficientRadius;
						float l0 = 3.75 / (4 * M_PI * 0.007439);
						float w02 = 1 - (1 - expf(-l / l0)) * 0.47; //1-0.025-0.0122*curEfficientRadius;//1-ka/ke;
						float w01 = 1.0;							//no absorbing for vis channel

						//Calculate scattering angle ��
						float theta = acos(-u * u0 + sin(sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + i * WIDTH + j) + 0]) * sin(satZenith_mat[WIDTH * i + j]) * cos(phi));

						//Calculate Henyey-Greenstein phase function from g1
						float phase1 = (1 - g1 * g1) / powf((1 + g1 * g1 - 2 * g1 * cos(theta)), 1.5);

						float A = 1.247;
						float B = 1.186;
						float C = 5.157;
						float RInf01 = (A + B * (u + u0) + C * u * u0 + phase1) / (4 * (u + u0));
						float visR = visData[nframe * HEIGHT * WIDTH + i * HEIGHT + j];
						//	   float t1=1.0/(K0u*K0u0/(RInf01-visData[nframe*512*512+i*512+j])-A1/(1-A1));
						float t1 = 1.0 / (K0u * K0u0 / (RInf01 - visData[nframe * HEIGHT * WIDTH + i * HEIGHT + j]));
						float vis = visData[nframe * HEIGHT * WIDTH + i * HEIGHT + j];
						float Thickness1 = 4 * (1.0 / t1 - 1.072) / (3 * (1 - g1));

						if (Thickness1 < 0 || Thickness1 > MAXVAL)
						{
							//cout<<"ice thickness too small or too greater:"<<endl;
							Thickness1 = 0.5;
						}

						//   float xi1=2*M_PI*curEfficientRadius/wavelength1;
						//  float xi2=2*M_PI*curEfficientRadius/wavelength2;
						float Thickness2 = Thickness1; //Thickness1*powf(wavelength2/wavelength1,2.0/3)*(1.1+powf(xi2,2.0/3))/(1.1+powf(xi1,2.0/3));

						float x2 = sqrtf(3 * (1 - g2) * (1 - w02)) * Thickness2;
						float y2 = 4 * sqrtf((1 - w02) / (3 * (1 - g2)));
						float tc = sinh(y2) / sinh(1.072 * y2 + x2);
						float t2 = tc - (4.86 - 13.08 * u * u0 + 12.76 * u * u * u0 * u0) * expf(x2) / powf(Thickness2, 3);
						float a2 = expf(-y2) - tc * expf(-x2 - y2);

						float phase2 = (1 - g2 * g2) / powf((1 + g2 * g2 - 2 * g2 * cos(theta)), 1.5);
						float RInf02 = (A + B * (u + u0) + C * u * u0 + phase2) / (4 * (u + u0));

						float mu = K0u0 * K0u / RInf02;
						//mu=mu*(1-0.05*y2);
						float curReflectance = RInf02 * expf(-y2 * mu) - (expf(-x2 - y2) - t2 * A2 / (1 - A2 * a2)) * t2 * K0u * K0u0;

						reflectanceSampleList[sample].refValue = curReflectance;
						reflectanceSampleList[sample].thick1 = Thickness1;
						reflectanceSampleList[sample].thick2 = Thickness2;
					}

					float minDif = 9999;
					int minSample = 0;
					for (int sample = 0; sample < sampleRes; sample++)
					{
						float visR = visData[nframe * WIDTH * HEIGHT + i * WIDTH + j];
						float irR = irReflectanceData[nframe * WIDTH * HEIGHT + i * WIDTH + j];
						float sampleIrR = reflectanceSampleList[sample].refValue;
						float dif = fabs(irR - sampleIrR);
						if (dif < minDif)
						{
							minDif = dif;
							minSample = sample;
						}
					}
					vis_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = reflectanceSampleList[minSample].thick1;
					ir4_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = reflectanceSampleList[minSample].thick2;
					reflectanceSampleList[minSample].radius = MinRadius + minSample * delta_radius;
					efficientRadius_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = reflectanceSampleList[minSample].radius;
					//	   cout<<"ice: Raidus: "<< reflectanceSampleList[minSample].radius<<endl;
				}
			}
		cout << nframe << endl;
	}
	delete[] reflectanceSampleList;
}

void SatDataCloud::ComputeGeoThick() //No.25
{
	PrintRunIfo("Geometric thickness");
	geo_thick_data = new float[NFRAME * WIDTH * HEIGHT];
	extinctionPlane = new float[WIDTH * HEIGHT * NFRAME];

	float max_cirrus_ext = -MAXVAL;
	float max_water_ext = -MAXVAL;
	float max_ice_ext = -MAXVAL;

	float max_cirrus_thick = -MAXVAL;
	float max_water_thick = -MAXVAL;
	float max_ice_thick = -MAXVAL;

	for (int nframe = 0; nframe < NFRAME; nframe++)
	{
		int exception_H_count = 0;

		float* satZenith_mat;
		float* satAzimuth_mat;
		if (nframe % 2)
		{
			satZenith_mat = satZenith_mat_F;
			satAzimuth_mat = satAzimuth_mat_F;
		}
		else
		{
			satZenith_mat = satZenith_mat_E;
			satAzimuth_mat = satAzimuth_mat_E;
		}
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 0;
				geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 0;

				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 1) //water cloud
				{
					float rho = 1.0e3;
					float wavelength1 = 0.7e-6;
					float wavelength2 = 3.75e-6;

					float re = efficientRadius_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] * 1.0e-6;
					float LWP = 2.0 / 3 * ir4_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] * rho * re / (1.0 + 1.1 / powf(2 * M_PI / wavelength2 * re, 2.0 / 3));
					float LWP2 = 2.0 / 3 * vis_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] * rho * re / (1.0 + 1.1 / powf(2 * M_PI / wavelength1 * re, 2.0 / 3));

					int N0 = 300e6;

					float alpha = 2;
					float rn = re / (alpha + 2);
					float V = 4 / 3.0 * M_PI * N0 * pow(rn, 3) * 24;
					float LWC = rho * V;

					float beta = 1.5 * LWC / (rho * re);

					extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j] = beta;
					float cth = cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float deltaZ = fabs(LWP / LWC);
					if (deltaZ > cth)
					{
						/*					cout<<deltaZ<<" :  water Thick too greater! "<<endl;*/
						exception_H_count++;
					}
					while (deltaZ > cth)
					{
						//deltaZ = cth * visData[nframe * WIDTH * HEIGHT + i * WIDTH + j];
						deltaZ *= 0.9;
					}
					//cout<<"water deltaz:  "<<deltaZ<<endl;
					geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = deltaZ;
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 8) //thin cirrus
				{
					float thicT = thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float irT = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float irWV = ir3Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					float emission = 0.0;
					emission = (PlanckFunction(0, ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j]) - PlanckFunction(0, ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j])) / (PlanckFunction(0, thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j]) - PlanckFunction(0, ground_temperature_mat[nframe * WIDTH * HEIGHT + i * WIDTH + j]));

					if (emission < 0 || emission > 1)
						emission = 0.6;

					float opticalDepth = -logf(1 - emission);
					opticalDepth *= fabs(cosf(satZenith_mat[WIDTH * i + j]));

					float deltaZ;
					float Tc = thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j] - 273;
					if (Tc < -70)
						Tc = -70;
					if (Tc > -10)
						Tc = -10;

					if (Tc < -35)
						deltaZ = 0.0456 * Tc + 4.7; //4.7
					else
						deltaZ = -0.065 * Tc + 0.725; //0.725;
					float beta = 2 * opticalDepth / (deltaZ * 1000);
					extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j] = beta;
					geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = deltaZ * 1000; //scale to 1/3
																							 //cout<<"cirrus deltaz:  "<<deltaZ<<endl;
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 3 || pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 2) //ice cloud
				{
					float re = efficientRadius_data[nframe * WIDTH * HEIGHT + i * WIDTH + j];

					float temp;
					if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 2)
						temp = thinCloudTList[nframe * WIDTH * HEIGHT + i * WIDTH + j];
					else
						temp = ir1Data[nframe * WIDTH * HEIGHT + i * WIDTH + j];

					if (temp > 253)
						temp = 253;
					float IWC = expf(-7.6 + 4 * expf(-0.2443 * 0.001 * powf(253 - temp, -2.4)));

					IWC = 0.3;
					float a0 = -6.656 * 0.001;
					float a1 = 3.686;
					float De_FL = 1.1 * re;
					float beta = IWC * (a0 + a1 / De_FL);
					beta = 0.0025;

					float deltaZ = vis_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] / beta;
					float cth = cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j];

					if (deltaZ > cth)
					{
						exception_H_count++;
					}
					while (deltaZ > cth)
					{
						//deltaZ = cth * visData[nframe * WIDTH * HEIGHT + i * WIDTH + j];
						deltaZ *= 0.9;
					}
					geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = deltaZ;
					extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j] = beta;
					/*						cout<<"ice deltaz:  "<<deltaZ<<endl;*/
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 0)
				{
					extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 0;
					geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] = 0;
				}
			}
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 1) //water cloud
				{
					max_water_thick = max(max_water_thick, geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
					max_water_ext = max(max_water_ext, extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 2) //thin cloud
				{
					max_cirrus_thick = max(max_cirrus_thick, geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
					max_cirrus_ext = max(max_cirrus_ext, extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
				}
				if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] == 3) //ice cloud
				{
					max_ice_thick = max(max_ice_thick, geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
					max_ice_ext = max(max_ice_ext, extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
				}
			}
		cout << "nframe: " << nframe << "Exception thickness count:  " << exception_H_count << endl;
		cout << "max water,thin, ice ext:  " << max_water_ext << "  , " << max_cirrus_ext << " , " << max_ice_ext << endl;
		cout << "max water,thin, ice thick:  " << max_water_thick << "  , " << max_cirrus_thick << " , " << max_ice_thick << endl;
	}
}

// 采样成粒子并保存
void SatDataCloud::GenerateCloudParticlesFile(string savePath, string saveName, int startFrame, int endFrame)     //No.42
{
	float min_radius = 0.12;

	float scale = 1;
	float geo_delta = 0;

	float  maxHeight = -MAXVAL;
	float  minHeight = MAXVAL;
	float minExt = MAXVAL;
	float maxExt = -MAXVAL;
	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				float cbh_scale;
				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] > 0)
				{
					float cth = cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					cbh_scale = cth - geo_delta - scale * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];

					maxHeight = max(maxHeight, cth);
					minHeight = min(minHeight, cbh_scale);

					minExt = min(minExt, extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j]);
					maxExt = max(maxExt, extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j]);
				}

			}
	}
	float  Lat_Ext = 60 * M_PI / 180;
	float  Log_Ext = 70 * M_PI / 180;
	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		puffPosVec.clear();
		puffColorVec.clear();
		puffSizeVec.clear();
		puffExtVec.clear();
		puffNumber = 0;
		for (int i = 0; i < HEIGHT; i++)
			for (int j = 0; j < WIDTH; j++)
			{
				float ph = -Log_Ext / 2 + j * Log_Ext / (WIDTH - 1);
				float th = Lat_Ext - i * Lat_Ext / (HEIGHT - 1);
				int  incre = 1;
				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] > 0)
				{

					float tempCTH = pow(cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j] - minHeight, incre) / pow(maxHeight - minHeight, incre) * 2;

					float cth = cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					float cbh_scale;
					cbh_scale = cth - geo_delta - scale * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];

					float tempCBH = pow(cbh_scale - minHeight, incre) / pow(maxHeight - minHeight, incre) * 2;

					float totalHeight = tempCTH - tempCBH;
					float  earth_radius = 50;

					if (totalHeight > F_ZERO)
					{
						float cur_h = tempCTH;
						int count = 10;
						while (cur_h > tempCBH && count--)
						{
							//float  z = (earth_radius + cur_h) * cosf(th) * cos(ph);
							//float  x = (earth_radius + cur_h) * cosf(th) * sin(ph);
							//float  y = (earth_radius + cur_h) * sinf(th);

							float x = -5 + min_radius * i;
							float y = -5 + min_radius * j;
							float z = 5 + cur_h;

							puffPosVec.push_back(Vector3(x, y, z));
							puffColorVec.push_back(Color4(1, 1, 1, 1));

							puffSizeVec.push_back(min_radius);
							float curExt = extinctionPlane[i * WIDTH + j] / maxExt * 30 + 60;
							puffExtVec.push_back(curExt);

							puffNumber++;
							cur_h -= min_radius;

						}
					}
				}//if
			}//for

		string partilcefileName;
		if ((endFrame - startFrame) <= 1)
		{
			partilcefileName = savePath + saveName + ".dat";
		}
		else
		{
			partilcefileName = savePath + saveName + to_string(nframe) + ".dat";
		}

		ExportCloudModel(const_cast<char*>(partilcefileName.c_str()));

		cout << "sampling file number: " << nframe << "Sampling Number:  " << puffNumber << endl;
	}
}

void SatDataCloud::ExportCloudModel(char* cloudfile)     //No.36
{
	FILE* fp = NULL;
	fp = fopen(cloudfile, "wb");
	if (!fp) return;

	fwrite(&puffNumber, sizeof(int), 1, fp);

	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffPosVec[i], sizeof(Vector3), 1, fp);

	}
	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffSizeVec[i], sizeof(float), 1, fp);
	}
	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffColorVec[i], sizeof(Color4), 1, fp);
	}
	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffExtVec[i], sizeof(float), 1, fp);
	}
	fclose(fp);
}


void SatDataCloud::GenerateVolumeFile(const string& savePath, int startFrame, int endFrame)
{
	//int zaxis = (this->HEIGHT + this->WIDTH) / 8;
	//int zaxis = (this->HEIGHT + this->WIDTH) / 2;
	int zaxis =50;

	float maxHeight = -MAXVAL;
	float minHeight = MAXVAL;
	float minExt = MAXVAL;
	float maxExt = -MAXVAL;

	float scale = 1;
	float geo_delta = 0;
	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				if (pixelTypeList[nframe * WIDTH * HEIGHT + WIDTH * i + j] > 0)
				{
					float cth = cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j];
					float cbh_scale = cth - geo_delta - scale * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];

					maxHeight = max(maxHeight, cth);
					minHeight = min(minHeight, cbh_scale);

					minExt = min(minExt, extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j]);
					maxExt = max(maxExt, extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j]);
				}

			}
		}
	}
	printf("MaxHeight: %f  -  MinHeight: %f  -  MaxExt: %f  -  MinExt: %f\n", maxHeight, minHeight, maxExt, minExt);
	float normHeight = maxHeight - minHeight;
	float normExt = maxExt - minExt;

	long long size = (long long)HEIGHT * (long long)WIDTH * (long long)zaxis;
	std:cout << "size :" << size << endl;
	vector<float> volumeData(size, 0);

	int count = 0;
	for (int nframe = startFrame; nframe < endFrame; nframe++)
	{
		for (int i = 0; i < HEIGHT; i++)
		{
			for (int j = 0; j < WIDTH; j++)
			{
				float cth = cthList[nframe * WIDTH * HEIGHT + WIDTH * i + j];
				float cbh_scale = cth - geo_delta - scale * geo_thick_data[nframe * WIDTH * HEIGHT + WIDTH * i + j];

				int qCth = min((int)(((cth - minHeight) / normHeight) * zaxis), zaxis);
				int qCbh = max((int)(((cbh_scale - minHeight) / normHeight) * zaxis), 0);

				//float ext = (extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j] - minExt) / normExt;
				float ext = extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j] / normExt;
				if(maxExt == minExt)
					ext = extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j];

				//if (qCbh <= qCth)
				//if(extinctionPlane[nframe * WIDTH * HEIGHT + WIDTH * i + j] > 0)
				if(ext > 0)
					count++;

				for (int k = qCbh; k <= qCth; k++)
				{
					long long idx1 = (long long)HEIGHT * (long long)WIDTH * (long long)zaxis * nframe;
					long long idx2 = (long long)WIDTH * (long long)zaxis * (long long)i;
					long long idx3 = (long long)zaxis * (long long)j;
					// std::cout << idx1 + idx2 + idx3 << " ";
					volumeData[idx1 + idx2 + idx3 + k] = ext;
				}
			}
		}
	}

	cout << "paint cloud nums: " << count << endl;

	string tmp = savePath;
	// WriteVTI(zaxis-1, WIDTH-1, HEIGHT-1, volumeData, tmp + "_ZWH.vti");
	WriteVTI(zaxis-1, WIDTH-1, HEIGHT-1, volumeData, tmp + ".vti");
}


/*
// // 将密度数据保存为.vti文件(ascii形式)(length，width，height：数据场长宽高；data：密度数据；path：文件保存路径)
// bool SatDataCloud::WriteVTI(int length, int width, int height, const std::vector<float>& data, std::string& path) {
//     std::ofstream file(path, std::ios::out | std::ios::trunc);
//     if (file) {
//         file << "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"";
//         // 判断大小端
//         int a = 1;
//         char* p = reinterpret_cast<char*>(&a);
//         if (*p == 1) {
//             file << "LittleEndian";
//         }
//         else {
//             file << "BigEndian";
//         }
//         file << "\" header_type=\"UInt64\">" << std::endl;

//         file << "<ImageData WholeExtent=\"" << "0 " << length << " 0 " << width << " 0 " << height
//             << "\" Origin=\"0 0 0\" Spacing=\"1.0 1.0 1.0\">" << std::endl;
//         file << "<Piece Extent=\"" << "0 " << length << " 0 " << width << " 0 " << height << "\">" << std::endl;
//         file << "<PointData Scalars=\"Scalars_\">" << std::endl;
//         float rangeMin = 1.0f;
//         float rangeMax = 0.0f;
//         for(float value:data) {
//             if(value < rangeMin) {
//                 rangeMin = value;
//             }
//             if(value > rangeMax) {
//                 rangeMax = value;
//             }
//         }
//         file << "<DataArray type=\"Float32\" Name=\"Scalars_\" format=\"ascii\" RangeMin=\""
//             << rangeMin << "\" RangeMax=\"" << rangeMax << "\">" << std::endl;
        
//         for (float value : data) {
//             file << value << " ";
//         }

//         file << "</DataArray>" << std::endl;
//         file << "</PointData>" << std::endl;
//         file << "<CellData>" << std::endl;
//         file << "</CellData>" << std::endl;
//         file << "</Piece>" << std::endl;
//         file << "</ImageData>" << std::endl;
//         file << "</VTKFile>" << std::endl;
//         file.close();

//     }
//     else {
//         printf("Fail to save vti file: %s!\n", path.c_str());
//         return false;
//     }
//     return true;
// }
*/

// Draw部分
/*
void SatDataCloud::Draw(DrawType type) //No.1
{
	extern int curFrame;
	switch (type)
	{
	case LongLat:
		DrawLongLatTable();
		break;
	case SunZA:
		DrawSunZenithAzimuth(curFrame);
		break;
	case Seg:
		DrawPixelType(curFrame);
		break;
	case TopSurface:
		DrawCloudTopSurface(curFrame);
	case BottomSurface:
		DrawCloudBottomSurface(curFrame);
		break;
	case Thick:
		DrawGeoThick(curFrame);
		break;
	case Extinction:
		DrawExt(curFrame);
		break;
	case EffRadius:
		DrawEfficientRadius(curFrame);
		break;
	default:
		break;
	}
}
void SatDataCloud::DrawPixelType(int nframe) //No.12
{
	if (pixelTypeList == NULL)
		return;
	glDisable(GL_LIGHTING);
	float *curData = new float[3 * WIDTH * HEIGHT];
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			switch (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j])
			{
			case 0:
				curData[3 * (i * WIDTH + j) + 0] = 0;
				curData[3 * (i * WIDTH + j) + 1] = 0.21;
				curData[3 * (i * WIDTH + j) + 2] = 0.40;
				break;
			case 1:
				curData[3 * (i * WIDTH + j) + 0] = 0.3;
				curData[3 * (i * WIDTH + j) + 1] = 0.3;
				curData[3 * (i * WIDTH + j) + 2] = 0.3;
				break;
			case 2:
				curData[3 * (i * WIDTH + j) + 0] = 0;
				curData[3 * (i * WIDTH + j) + 1] = 0.6;
				curData[3 * (i * WIDTH + j) + 2] = 0.0;
				break;
			case 3:
				curData[3 * (i * WIDTH + j) + 0] = 0.8;
				curData[3 * (i * WIDTH + j) + 1] = 0.8;
				curData[3 * (i * WIDTH + j) + 2] = 0.8;
				break;
			case 4:
				curData[3 * (i * WIDTH + j) + 0] = 0.0;
				curData[3 * (i * WIDTH + j) + 1] = 1.0;
				curData[3 * (i * WIDTH + j) + 2] = 0.0;
				break;
			case 5:
				curData[3 * (i * WIDTH + j) + 0] = 0.0;
				curData[3 * (i * WIDTH + j) + 1] = 0.0;
				curData[3 * (i * WIDTH + j) + 2] = 1.0;
				break;
			default:
				curData[3 * (i * WIDTH + j) + 0] = 1.0;
				curData[3 * (i * WIDTH + j) + 1] = 1.0;
				curData[3 * (i * WIDTH + j) + 2] = 0.0;
				break;
			}
		}
	for (int i = 0; i < HEIGHT - 1; i++)
		for (int j = 0; j < WIDTH - 1; j++)
		{
			glBegin(GL_POLYGON);
			glColor3fv(curData + 3 * (i * WIDTH + j));
			glVertex2f(float(j) / WIDTH, 1.0 - float(i) / HEIGHT);
			glColor3fv(curData + 3 * ((i + 1) * WIDTH + j));
			glVertex2f(float(j) / WIDTH, 1.0 - float(i + 1) / HEIGHT);
			glColor3fv(curData + 3 * ((i + 1) * WIDTH + j + 1));
			glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i + 1) / HEIGHT);
			glColor3fv(curData + 3 * (i * WIDTH + j + 1));
			glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i) / HEIGHT);
			glEnd();
		}
	delete[] curData;
}
void SatDataCloud::DrawCloudTopSurface(int nframe) //No.16
{
	if (cthList == NULL)
		return;
	glEnable(GL_LIGHTING);
	float *curData = new float[WIDTH * HEIGHT];
	float maxData = -99999;
	float minData = 99999;
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				curData[i * WIDTH + j] = cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] / SCALE;
				if (maxData < curData[i * WIDTH + j])
					maxData = curData[i * WIDTH + j];
				if (minData > curData[i * WIDTH + j])
					minData = curData[i * WIDTH + j];
			}
		}
	//cloud top surface
	for (int i = 0; i < HEIGHT - 1; i++)
		for (int j = 0; j < WIDTH - 1; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + (i + 1) * WIDTH + j] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + (i + 1) * WIDTH + j + 1] != 0)
			{
				double normal[3], PA[3], PB[3], PC[3];
				PA[0] = j * 1.0 / WIDTH;
				PA[1] = (HEIGHT - i) * 1.0 / HEIGHT;
				PA[2] = curData[i * WIDTH + j];
				PB[0] = (j)*1.0 / WIDTH;
				PB[1] = (HEIGHT - i - 1) * 1.0 / HEIGHT;
				PB[2] = curData[(i + 1) * WIDTH + j];
				PC[0] = (j + 1) * 1.0 / WIDTH;
				PC[1] = (HEIGHT - (i + 1)) * 1.0 / HEIGHT;
				PC[2] = curData[(i + 1) * WIDTH + j + 1];
				ComputeTriangleNormal(normal, PA, PB, PC);
				glNormal3dv(normal);
				glBegin(GL_TRIANGLES);
				glVertex3dv(PA);
				glVertex3dv(PB);
				glVertex3dv(PC);
				glEnd();
			}
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j + 1] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + (i + 1) * WIDTH + j + 1] != 0)
			{
				double normal[3], PA[3], PC[3], PD[3];
				PA[0] = j * 1.0 / WIDTH;
				PA[1] = (HEIGHT - i) * 1.0 / HEIGHT;
				PA[2] = curData[i * WIDTH + j];
				PC[0] = (j + 1) * 1.0 / WIDTH;
				PC[1] = (HEIGHT - (i + 1)) * 1.0 / HEIGHT;
				PC[2] = curData[(i + 1) * WIDTH + j + 1];
				PD[0] = (j + 1) * 1.0 / WIDTH;
				PD[1] = (HEIGHT - i) * 1.0 / HEIGHT;
				PD[2] = curData[i * WIDTH + j + 1];
				ComputeTriangleNormal(normal, PA, PC, PD);
				glNormal3dv(normal);
				glBegin(GL_TRIANGLES);
				glVertex3dv(PA);
				glVertex3dv(PC);
				glVertex3dv(PD);
				glEnd();
			}
		}
	delete[] curData;
}
void SatDataCloud::DrawEfficientRadius(int nframe) //No.24
{
	if (efficientRadius_data == NULL)
		return;
	glDisable(GL_LIGHTING);
	float *curData = new float[WIDTH * HEIGHT];
	float maxData = -99999;
	float minData = 99999;
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			maxData = max(maxData, efficientRadius_data[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
		}
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			curData[i * WIDTH + j] = efficientRadius_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] / maxData;
		}
	for (int i = 0; i < HEIGHT - 1; i++)
		for (int j = 0; j < WIDTH - 1; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				glBegin(GL_QUADS);
				glColor3f(curData[i * WIDTH + j], curData[i * WIDTH + j], curData[i * WIDTH + j]);
				glVertex2f(float(j) / WIDTH, 1.0 - float(i) / HEIGHT);
				glColor3f(curData[(i + 1) * WIDTH + j], curData[(i + 1) * WIDTH + j], curData[(i + 1) * WIDTH + j]);
				glVertex2f(float(j) / WIDTH, 1.0 - float((i + 1)) / HEIGHT);
				glColor3f(curData[(i + 1) * WIDTH + j + 1], curData[(i + 1) * WIDTH + j + 1], curData[(i + 1) * WIDTH + j + 1]);
				glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i + 1) / HEIGHT);
				glColor3f(curData[i * WIDTH + j + 1], curData[i * WIDTH + j + 1], curData[i * WIDTH + j + 1]);
				glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i) / HEIGHT);
				glEnd();
			}
		}
	delete[] curData;
}
void SatDataCloud::DrawCloudBottomSurface(int nframe) //No.27
{
	//cloud bottom surface
	if (cthList == NULL || geo_thick_data == NULL)
		return;
	glEnable(GL_LIGHTING);
	float *curData = new float[WIDTH * HEIGHT];
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float temp1 = cthList[nframe * WIDTH * HEIGHT + i * WIDTH + j] / SCALE;
			float temp2 = fabs(geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j]) / SCALE;
			float temp3 = temp1 - temp2;
			curData[i * WIDTH + j] = temp3;
		}
	for (int i = 0; i < HEIGHT - 1; i++)
		for (int j = 0; j < WIDTH - 1; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + (i + 1) * WIDTH + j] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + (i + 1) * WIDTH + j + 1] != 0)
			{
				double normal[3], PA[3], PB[3], PC[3];
				PA[0] = j * 1.0 / WIDTH;
				PA[1] = (HEIGHT - i) * 1.0 / HEIGHT;
				PA[2] = curData[i * WIDTH + j];
				PB[0] = (j)*1.0 / WIDTH;
				PB[1] = (HEIGHT - i - 1) * 1.0 / HEIGHT;
				PB[2] = curData[(i + 1) * WIDTH + j];
				PC[0] = (j + 1) * 1.0 / WIDTH;
				PC[1] = (HEIGHT - (i + 1)) * 1.0 / HEIGHT;
				PC[2] = curData[(i + 1) * WIDTH + j + 1];
				ComputeTriangleNormal(normal, PA, PC, PB);
				glNormal3dv(normal);
				glBegin(GL_TRIANGLES);
				glVertex3dv(PA);
				glVertex3dv(PC);
				glVertex3dv(PB);
				glEnd();
			}
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j + 1] != 0 && pixelTypeList[nframe * WIDTH * HEIGHT + (i + 1) * WIDTH + j + 1] != 0)
			{
				double normal[3], PA[3], PC[3], PD[3];
				PA[0] = j * 1.0 / WIDTH;
				PA[1] = (HEIGHT - i) * 1.0 / HEIGHT;
				PA[2] = curData[i * WIDTH + j];
				PC[0] = (j + 1) * 1.0 / WIDTH;
				PC[1] = (HEIGHT - (i + 1)) * 1.0 / HEIGHT;
				PC[2] = curData[(i + 1) * WIDTH + j + 1];
				PD[0] = (j + 1) * 1.0 / WIDTH;
				PD[1] = (HEIGHT - i) * 1.0 / HEIGHT;
				PD[2] = curData[i * WIDTH + j + 1];
				ComputeTriangleNormal(normal, PA, PD, PC);
				glNormal3dv(normal);
				glBegin(GL_TRIANGLES);
				glVertex3dv(PA);
				glVertex3dv(PD);
				glVertex3dv(PC);
				glEnd();
			}
		}
	delete[] curData;
}
void SatDataCloud::ComputeTriangleNormal(double normal[3], double PA[3], double PB[3], double PC[3]) //No.17
{
	double vecAB[3], vecAC[3];
	for (int i = 0; i < 3; i++)
	{
		vecAB[i] = PB[i] - PA[i];
		vecAC[i] = PC[i] - PA[i];
	}
	normal[0] = vecAB[1] * vecAC[2] - vecAB[2] * vecAC[1];
	normal[1] = vecAB[2] * vecAC[0] - vecAB[0] * vecAC[2];
	normal[2] = vecAB[0] * vecAC[1] - vecAB[1] * vecAC[0];
	double len = sqrt(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
	for (int i = 0; i < 3; i++)
	{
		normal[i] /= len;
	}
}
void SatDataCloud::DrawLongLatTable() //No.37
{
	if (longitudeLatitudeTable == NULL)
		return;
	float minLg = MAXVAL;
	float maxLg = -MAXVAL;
	float minLat = MAXVAL;
	float maxLat = -MAXVAL;
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float lg = longitudeLatitudeTable[2 * (i * WIDTH + j) + 0];
			float lat = longitudeLatitudeTable[2 * (i * WIDTH + j) + 1];
			minLat = min(minLat, lat);
			maxLat = max(maxLat, lat);
			minLg = min(minLg, lg);
			maxLg = max(maxLg, lg);
		}
	glBegin(GL_POINTS);
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float lg = longitudeLatitudeTable[2 * (i * WIDTH + j) + 0];
			float lat = longitudeLatitudeTable[2 * (i * WIDTH + j) + 1];
			float z = cosf(lat) * cosf(lg);
			float x = cosf(lat) * sinf(lg);
			float y = sinf(lat);
			glColor3f((lat - minLat) / (maxLat - minLat), (lg - minLg) / (maxLg - minLg) / 2.0, 0.0);
			glVertex3f(x, y, z);
		}
	glEnd();
	DrawEarth();
}
void SatDataCloud::DrawSunZenithAzimuth(int nframe) //No.48
{
	float minA = MAXVAL;
	float maxA = -MAXVAL;
	float minZ = MAXVAL;
	float maxZ = -MAXVAL;
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float sunZ = sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + WIDTH * i + j) + 0];
			float sunA = sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + WIDTH * i + j) + 1];
			minA = min(minA, sunA);
			maxA = max(maxA, sunA);
			minZ = min(minZ, sunZ);
			maxZ = max(maxZ, sunZ);
		}
	glEnd();
	glBegin(GL_POINTS);
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			float lg = longitudeLatitudeTable[2 * (i * WIDTH + j) + 0];
			float lat = M_PI / 2 - longitudeLatitudeTable[2 * (i * WIDTH + j) + 1];
			float z = sinf(lat) * cosf(lg);
			float x = sinf(lat) * sinf(lg);
			float y = cos(lat);
			float sunZ = sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + WIDTH * i + j) + 0];
			float sunA = sunZenithAzimuth_mat[2 * (nframe * WIDTH * HEIGHT + WIDTH * i + j) + 1];
			float scale = 1.0 - (sunZ - minZ) / (maxZ - minZ);
			glColor3f(scale, scale, scale);
			glVertex3f(1.2 * x, 1.2 * y, 1.2 * z);
		}
	glEnd();
	DrawEarth();
}
void SatDataCloud::DrawEarth() //No.38
{
	glDisable(GL_LIGHTING);
	float *earth_texture = new float[1441 * 721 * 3];
	FILE *fp = NULL;
	fp = fopen("earth_texture.dat", "rb"); //read earth texture
	fread(earth_texture, sizeof(float), 1441 * 721 * 3, fp);
	fclose(fp);
	float dthta = 0.25;
	float dphi = 0.25;
	glBegin(GL_QUADS);
	for (int i = 0; i < 721 - 1; i++)
		for (int j = 0; j < 1441 - 1; j++)
		{
			float x, y, z;
			float th, ph;
			th = i * dthta * M_PI / 180;
			ph = (j * dphi) * M_PI / 180 - M_PI;
			z = sin(th) * cos(ph);
			x = sin(th) * sin(ph);
			y = cos(th);
			glColor3fv(earth_texture + i * 1441 * 3 + j * 3);
			glVertex3f(x, y, z);
			th = (i + 1) * dthta * M_PI / 180;
			ph = j * dphi * M_PI / 180 - M_PI;
			z = sin(th) * cos(ph);
			x = sin(th) * sin(ph);
			y = cos(th);
			glColor3fv(earth_texture + (i + 1) * 1441 * 3 + j * 3);
			glVertex3f(x, y, z);
			th = (i + 1) * dthta * M_PI / 180;
			ph = (j + 1) * dphi * M_PI / 180 - M_PI;
			z = sin(th) * cos(ph);
			x = sin(th) * sin(ph);
			y = cos(th);
			glColor3fv(earth_texture + (i + 1) * 1441 * 3 + (j + 1) * 3);
			glVertex3f(x, y, z);
			th = i * dthta * M_PI / 180;
			ph = (j + 1) * dphi * M_PI / 180 - M_PI;
			z = sin(th) * cos(ph);
			x = sin(th) * sin(ph);
			y = cos(th);
			glColor3fv(earth_texture + i * 1441 * 3 + (j + 1) * 3);
			glVertex3f(x, y, z);
		}
	glEnd();
	glBegin(GL_POINTS);
	glColor3f(1, 1, 1);
	for (int i = 0; i < 512; i++)
	{
		float lg = 0.0;
		float lat = i * M_PI / 511;
		float z = sinf(lat) * cosf(lg);
		float x = sinf(lat) * sinf(lg);
		float y = cos(lat);
		glVertex3f(x, y, z);
		//glVertex3f(lg/M_PI,lat/M_PI,1.0);
	}
	glEnd();
	glBegin(GL_POINTS);
	glColor3f(0, 1, 0);
	for (int i = 0; i < 512; i++)
	{
		float lg = M_PI;
		float lat = i * M_PI / 511;
		float z = sinf(lat) * cosf(lg);
		float x = sinf(lat) * sinf(lg);
		float y = cos(lat);
		glVertex3f(x, y, z);
		//glVertex3f(lg/M_PI,lat/M_PI,1.0);
	}
	glEnd();
	glBegin(GL_POINTS);
	glColor3f(0, 1, 1);
	for (int i = 0; i < 512; i++)
	{
		float lg = -M_PI + i * 2 * M_PI / 511;
		float lat = M_PI / 2;
		float z = sinf(lat) * cosf(lg);
		float x = sinf(lat) * sinf(lg);
		float y = cos(lat);
		glVertex3f(x, y, z);
		//glVertex3f(lg/M_PI,lat/M_PI,1.0);
	}
	glEnd();
	//satellite
	float lg = 105 * M_PI / 180; //degree
	float lat = M_PI / 2;
	glPushMatrix();
	float z = 2 * sinf(lat) * cosf(lg);
	float x = 2 * sinf(lat) * sinf(lg);
	float y = 2 * cos(lat);
	glTranslatef(x, y, z);
	glutSolidSphere(0.01, 50, 50);
	glPopMatrix();
	delete[] earth_texture;
}
void SatDataCloud::DrawGeoThick(int nframe) //No.26
{
	if (geo_thick_data == NULL || nframe >= NFRAME)
		return;
	glDisable(GL_LIGHTING);
	float *curData = new float[WIDTH * HEIGHT];
	float maxData = -MAXVAL;
	float minData = MAXVAL;
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				maxData = max(maxData, geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
			}
		}
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				curData[i * WIDTH + j] = 0.2 + geo_thick_data[nframe * WIDTH * HEIGHT + i * WIDTH + j] / maxData;
			}
			else
			{
				curData[i * WIDTH + j] = 0.0;
			}
		}
	for (int i = 0; i < HEIGHT - 1; i++)
		for (int j = 0; j < WIDTH - 1; j++)
		{
			glBegin(GL_QUADS);
			glColor3f(curData[i * WIDTH + j], curData[i * WIDTH + j], curData[i * WIDTH + j]);
			glVertex2f(float(j) / WIDTH, 1.0 - float(i) / HEIGHT);
			glColor3f(curData[(i + 1) * WIDTH + j], curData[(i + 1) * WIDTH + j], curData[(i + 1) * WIDTH + j]);
			glVertex2f(float(j) / WIDTH, 1.0 - float((i + 1)) / HEIGHT);
			glColor3f(curData[(i + 1) * WIDTH + j + 1], curData[(i + 1) * WIDTH + j + 1], curData[(i + 1) * WIDTH + j + 1]);
			glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i + 1) / HEIGHT);
			glColor3f(curData[i * WIDTH + j + 1], curData[i * WIDTH + j + 1], curData[i * WIDTH + j + 1]);
			glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i) / HEIGHT);
			glEnd();
		}
	delete[] curData;
}
void SatDataCloud::DrawExt(int nframe) //No.58
{
	if (extinctionPlane == NULL)
		return;
	glDisable(GL_LIGHTING);
	float *curData = new float[WIDTH * HEIGHT];
	float maxData = -MAXVAL;
	float minData = MAXVAL;
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				maxData = max(maxData, extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j]);
			}
		}
	for (int i = 0; i < HEIGHT; i++)
		for (int j = 0; j < WIDTH; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				curData[i * WIDTH + j] = extinctionPlane[nframe * WIDTH * HEIGHT + i * WIDTH + j] / maxData;
			}
		}
	for (int i = 0; i < HEIGHT - 1; i++)
		for (int j = 0; j < WIDTH - 1; j++)
		{
			if (pixelTypeList[nframe * WIDTH * HEIGHT + i * WIDTH + j] > 0)
			{
				glBegin(GL_QUADS);
				glColor3f(curData[i * WIDTH + j], curData[i * WIDTH + j], curData[i * WIDTH + j]);
				glVertex2f(float(j) / WIDTH, 1.0 - float(i) / HEIGHT);
				glColor3f(curData[(i + 1) * WIDTH + j], curData[(i + 1) * WIDTH + j], curData[(i + 1) * WIDTH + j]);
				glVertex2f(float(j) / WIDTH, 1.0 - float((i + 1)) / HEIGHT);
				glColor3f(curData[(i + 1) * WIDTH + j + 1], curData[(i + 1) * WIDTH + j + 1], curData[(i + 1) * WIDTH + j + 1]);
				glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i + 1) / HEIGHT);
				glColor3f(curData[i * WIDTH + j + 1], curData[i * WIDTH + j + 1], curData[i * WIDTH + j + 1]);
				glVertex2f(float(j + 1) / WIDTH, 1.0 - float(i) / HEIGHT);
				glEnd();
			}
		}
	delete[] curData;
}
*/
// PrintRunIfo
void SatDataCloud::PrintRunIfo(string ifo) //No.59
{
	cout << ifo << "......" << endl;
}