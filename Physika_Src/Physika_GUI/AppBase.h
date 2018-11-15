#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <memory>
#include "Physika_Framework/Framework/SceneGraph.h"

namespace Physika
{
	class AppBase {
	public:
		AppBase(void) {};
		~AppBase() {};

		virtual void createWindow(int width, int height) {};
		virtual void mainLoop() = 0;
	};

}
