#pragma once
#include"element.h"
#include"node.h"

class DisCreteElement:public Element
{
	int nodeId_[2];
	NodeCuda node_[2];
};