#ifndef FRAMEWORK_ANIMATE_H
#define FRAMEWORK_ANIMATE_H

#include "Action.h"

namespace PhysIKA
{
	class AnimateAct : public Action
	{
	public:
		AnimateAct();
		virtual ~AnimateAct();

	private:
		void process(Node* node) override;
	};
}

#endif
