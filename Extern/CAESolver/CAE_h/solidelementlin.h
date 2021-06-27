#pragma once
#include"element.h"
#include"node.h"

class BaseSolidElementLin:public Element
{
	int nodeId_[8];
	NodeCuda node_[8];
};