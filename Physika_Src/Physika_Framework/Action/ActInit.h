#pragma once
#include "Action.h"

namespace Physika
{
	class InitAct : public Action
	{
	public:
		InitAct();
		virtual ~InitAct();

	private:
		void Process(Node* node) override;
	};
}
