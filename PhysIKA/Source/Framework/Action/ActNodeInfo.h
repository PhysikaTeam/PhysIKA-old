#ifndef FRAMEWORK_NODEINFOACT_H
#define FRAMEWORK_NODEINFOACT_H

#include "Action.h"

namespace PhysIKA
{
	class NodeInfoAct : public Action
	{
	public:
		NodeInfoAct();
		virtual ~NodeInfoAct();

	private:
		void process(Node* node) override;
	};
}

#endif
