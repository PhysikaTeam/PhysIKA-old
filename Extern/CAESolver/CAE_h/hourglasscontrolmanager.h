#pragma once
#include"hourglasscontrol.h"
//#include"book.h"
#include"helper_cuda.h"
#include<map>
using std::map;

//当前只支持整个问题域使用一种沙漏控制

typedef struct HourglassControlManager
{
	//cpu端
	vector<HourglassControl> hourglass_array_cpu;

	//gpu端
	HourglassControl* hourglass_array_gpu;

	map<int, HourglassControl*> hourglass_map;

	HourglassControl* returnHourglass(const int hg_id);

	HourglassControlManager();

	~HourglassControlManager()
	{
		if (hourglass_array_gpu) 
		{
			checkCudaErrors(cudaFree(hourglass_array_gpu));
		}
		hourglass_array_gpu = nullptr;
	}

	void createGpuHourglassArray();

	void createHourglassMap();

}HourglassControlManager;