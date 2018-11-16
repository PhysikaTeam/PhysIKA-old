#pragma once
#include "Physika_Framework/Framework/ModuleController.h"
#include "Physika_Core/Platform.h"

namespace Physika
{

class AnimationController : public ControllerModule
{
	DECLARE_CLASS(AnimationController)

public:
	AnimationController();
	virtual ~AnimationController();

	bool execute() override;
private:

};

}
