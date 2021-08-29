#pragma once
#include<string>

//ls:2020-03-17
struct Temp_transfor;

typedef struct Def_transfor
{
	int transform_id;
	//string option;
	//double A[7];
	vector<Temp_transfor>temp_transfor_array;

} Def_transfor;

typedef struct Temp_transfor
{
	string option;
	double A[7];

}Temp_transfor;