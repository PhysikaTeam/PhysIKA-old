#include"SatDataCloud_Lite.h"
#include"string"

int main()
{   
    SatDataCloud typhoon;
    // 参数分别为 - 定位输入文件的日期 - 输入文件路径 - 输出文件路径 - 输出文件名 - 提取台风的长度 - 提取台风的宽度
	//typhoon.Run(Date(2013,6,26), "D:\\Code\\PhysIKA_Cloud_Data\\Typhoon\\SULI\\SEC\\", ".\\", "typhoon20130626_50", 512, 512);

    vector<std::string> files;
	//files.emplace_back("D:\\Code\\PhysIKA_Cloud_Data\\Typhoon\\SULI\\SEC\\FY2E_SEC_VIS_MLS_20130625_0500.AWX");
 //   files.emplace_back("D:\\Code\\PhysIKA_Cloud_Data\\Typhoon\\SULI\\SEC\\FY2E_SEC_IR1_MLS_20130625_0500.AWX");
 //   files.emplace_back("D:\\Code\\PhysIKA_Cloud_Data\\Typhoon\\SULI\\SEC\\FY2E_SEC_IR2_MLS_20130625_0500.AWX");
 //   files.emplace_back("D:\\Code\\PhysIKA_Cloud_Data\\Typhoon\\SULI\\SEC\\FY2E_SEC_IR3_MLS_20130625_0500.AWX");
 //   files.emplace_back("D:\\Code\\PhysIKA_Cloud_Data\\Typhoon\\SULI\\SEC\\FY2E_SEC_IR4_MLS_20130625_0500.AWX");

	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C01_G16_s20172531622545_e20172531623002_c20172531623045.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C14_G16_s20172531622545_e20172531623002_c20172531623045.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C15_G16_s20172531622545_e20172531623008_c20172531623046.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C09_G16_s20172531622545_e20172531623008_c20172531623044.AWX");
	files.emplace_back("D:\\PhysIKA_Cloud_Data\\AWX_Data\\M1-2531622\\OR_ABI-L1b-RadM1-M3C07_G16_s20172531622545_e20172531623013_c20172531623043.AWX");

	// 参数分别为 - 文件列表 - 输出文件路径 -输出文件名 - 提取台风的长度 - 提取台风的宽度
    typhoon.Run(files, ".\\", "M1-2531622", 500, 500);

    return 0;
}