#pragma once
#include "element.h"
#include "node.h"

class BaseSolidElementQuad:public Element
{
	int nodeId_[20];
	NodeCuda node_[20];
};