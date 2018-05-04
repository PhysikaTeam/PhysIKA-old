#ifndef FRAMEWORK_NODEINFOACT_H
#define FRAMEWORK_NODEINFOACT_H

#include "Action.h"

namespace Physika
{
	class NodeInfoAct : public Action
	{
	public:
		NodeInfoAct();
		virtual ~NodeInfoAct();

	private:
		void Process(Node* node) override;
	};
}

#endif
