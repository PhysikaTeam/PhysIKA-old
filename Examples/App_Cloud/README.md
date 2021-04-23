# physIKA_Cloud

1. 请一定使用64位编译器；
2. 请一定在Windows平台下使用；

3. Configure选项：
	1. CLOUD_EULER  -  欧拉法
	2. CLOUD_PYTHON  -  python接口
	3. CLOUD_IMG_CUMULUS  -  RGB图像生成积云
	4. CLOUD_SAT_CUMULUS  -  卫星云图生成积云（不在项目计划内）
	5. CLOUD_SAT_TYPHOON  -  卫星云图生成台风
	6. CLOUD_WRF  -  WRF数据生成台风
	7. EXAMPLE_SATIMG_TYPHOON  -  卫星云图生成台风示例（仅在勾选CLOUD_SAT_TYPHOON下勾选此项）
	8. EXAMPLE_SATIMG_CUMULUS  -  卫星云图生成积云示例（仅在勾选CLOUD_SAT_CUMULUS下勾选此项）（不在项目计划内）
	9. EXAMPLE_WRF  -  WRF示例（仅在勾选CLOUD_WRF下勾选此项）
	10. USE_PREBUILT_OSG  -  是否使用预建的OSG（仅在Windows、Debug、VS2017以后版本可用，如果不满足条件请勿勾选此项）

4. 目录结构：
	1. Cplusplus    C++源码部分
	2. Examples    示例代码部分, 示意各部分如何使用
	3. Python    python接口代码部分

5. BUILD完成后一定INSTALL以自动拷贝动态链接库；

6. python接口使用方式详见各txt文件