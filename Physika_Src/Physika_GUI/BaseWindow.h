#pragma once
#include <vector>
#include <algorithm>
#include <iostream>

#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Framework/Framework/ModuleVisual.h"

namespace Physika
{
	class BaseWindow {
	public:
		BaseWindow(void) {};
		~BaseWindow() {};

		void setScene(std::shared_ptr<SceneGraph> root) { m_root = root; }
		std::shared_ptr<SceneGraph> getScene() { return m_root; }

		void drawScene() {
		}

	protected:
		std::shared_ptr<SceneGraph> m_root;
	};

}
