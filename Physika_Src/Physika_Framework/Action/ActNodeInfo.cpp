#include "ActNodeInfo.h"

namespace Physika
{
	
	NodeInfoAct::NodeInfoAct()
	{

	}

	NodeInfoAct::~NodeInfoAct()
	{

	}

	void NodeInfoAct::Process(Node* node)
	{
		std::cout << node->getName() << std::endl;
	}

}