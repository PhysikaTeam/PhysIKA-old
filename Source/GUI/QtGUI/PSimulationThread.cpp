#include "PSimulationThread.h"

#include "Framework/SceneGraph.h"

namespace PhysIKA
{
	PSimulationThread::PSimulationThread()
	{

	}

	void PSimulationThread::run()
	{
		SceneGraph::getInstance().initialize();

		while (true)
		{
			SceneGraph::getInstance().takeOneFrame();
		}
	}

}
