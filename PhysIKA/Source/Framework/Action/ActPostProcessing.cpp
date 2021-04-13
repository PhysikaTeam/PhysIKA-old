#include "ActPostProcessing.h"

namespace PhysIKA
{
	
	PostProcessing::PostProcessing()
	{

	}

	PostProcessing::~PostProcessing()
	{

	}

	void PostProcessing::process(Node* node)
	{
		auto mList = node->getModuleList();
		int cnt = 0;
		for (auto iter = mList.begin(); iter != mList.end(); iter++)
		{
			//printf("iter %s ::: %s\n", (*iter)->getName(),(*iter)->getModuleType());
			if (std::string("IOModule").compare((*iter)->getModuleType()) == 0)
			{
				
				(*iter)->execute();
			}
		}
	}
}