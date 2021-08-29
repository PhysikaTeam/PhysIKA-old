#pragma once
#include<vector>
#include"inc_transfor.h"

using std::vector;

struct Inc_transfor;

//ls:2020-03-17
typedef struct Inc_transforManager
{
	vector<Inc_transfor> inc_transfor_array;
} Inc_transforManager;
//