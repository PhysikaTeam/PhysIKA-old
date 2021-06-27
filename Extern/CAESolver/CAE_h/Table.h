#pragma once
#include <string>
using namespace std;

#define curve_max_node_num 128

//ls:2020-03-17
typedef struct Table
{
	string title;
	int node_num;
	double x_axis[curve_max_node_num];
	double y_axis[curve_max_node_num];
	int table_id;
	double SFA;//Scale factor for value.
	double SFO;//Scale factor for value.
	double OFFA;//Offset for abscissa values, see explanation below
}Table;
//