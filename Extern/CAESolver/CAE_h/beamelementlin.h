#pragma once
#include"element.h"
#include"structure_declaration.h"
#include"node.h"

class BaseBeamElementLin:public Element
{
	int nodeId_[2];
	NodeCuda node_[2];
};