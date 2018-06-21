#include "ActAnimate.h"

namespace Physika
{
	
	AnimateAct::AnimateAct()
	{

	}

	AnimateAct::~AnimateAct()
	{

	}

	void AnimateAct::Process(Node* node)
	{
		node->advance(node->getDt());
	}

}