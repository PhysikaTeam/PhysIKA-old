#pragma once
#include"element.h"
#include"node.h"
//ls:2020-03-17 (添加此类，待修改)


struct NodeCuda;
struct Element;

typedef struct MassElementCuda :Element
{
	int node_id_array[3];
	NodeCuda* node_array[3];

	//待修改
	int partset_id;
	int manager_id;
	int elm_id;
	int node_id;
	double addMass;
	double finalMass;

}MassElementCuda;