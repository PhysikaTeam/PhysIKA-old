#pragma once
#include "element.h"
#include"node.h"

class BaseShellElementLin :public Element
{
	int nodeId_[4];
	NodeCuda node_[4];
};