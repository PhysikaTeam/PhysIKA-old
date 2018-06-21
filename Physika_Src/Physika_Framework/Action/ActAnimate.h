#ifndef FRAMEWORK_ANIMATE_H
#define FRAMEWORK_ANIMATE_H

#include "Action.h"

namespace Physika
{
	class AnimateAct : public Action
	{
	public:
		AnimateAct();
		virtual ~AnimateAct();

	private:
		void Process(Node* node) override;
	};
}

#endif
