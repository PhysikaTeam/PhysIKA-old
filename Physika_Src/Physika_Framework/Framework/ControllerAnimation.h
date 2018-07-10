#pragma once
#include "Framework/ModuleController.h"
#include "Physika_Core/Platform.h"

namespace Physika
{

class AnimationController : public ControllerModule
{
	DECLARE_CLASS(AnimationController)

public:
	AnimationController();
	virtual ~AnimationController();

	virtual void step(Real dt);
private:

};

}
