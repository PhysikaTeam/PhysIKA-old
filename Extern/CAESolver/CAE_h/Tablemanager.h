#pragma once
#include<vector>
#include"Table.h"

using std::vector;

struct Table;

//ls:2020-03-17
typedef struct  TableManager
{
	vector<Table> table_array;
	vector<double> tableValue;
	Table *table_array_gpu;
	int totTableNum;

}TableManager;
//