#pragma once
#include "ElementMass.h"

struct ElementMassCuda;


typedef struct ElementMassManager
{
	vector<MassElementCuda> element_mass_array;
	int elm_num;

	//ElementMassManager();
	///**
	//将质量附加到节点上
	//*/
	//void addMassToNode();

	///**
	//连接节点
	//*/
	//void linkNode(NodeManager *nodeManager);
}ElementMassManager;