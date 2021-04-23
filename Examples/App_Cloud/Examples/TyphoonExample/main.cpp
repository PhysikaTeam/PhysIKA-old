#include"SatDataCloud_Lite.h"
#include"string"

int main()
{   
    SatDataCloud typhoon;

    vector<std::string> files;

	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C01_G16_s20172531622545_e20172531623002_c20172531623045.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C14_G16_s20172531622545_e20172531623002_c20172531623045.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C15_G16_s20172531622545_e20172531623008_c20172531623046.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C09_G16_s20172531622545_e20172531623008_c20172531623044.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C07_G16_s20172531622545_e20172531623013_c20172531623043.AWX");

	// 参数分别为 - 文件列表 - 输出文件路径 -输出文件名 - 提取台风的长度 - 提取台风的宽度
    typhoon.Run(files, ".\\", "M1-2531622", 500, 500);

    return 0;
}