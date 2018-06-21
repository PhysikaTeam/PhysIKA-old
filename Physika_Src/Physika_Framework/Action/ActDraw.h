#ifndef FRAMEWORK_NODEINFOACT_H
#define FRAMEWORK_NODEINFOACT_H

#include "Action.h"

namespace Physika
{
	class DrawAct : public Action
	{
	public:
		DrawAct();
		virtual ~DrawAct();

	private:
		void Process(Node* node) override;
	};
}

#endif
