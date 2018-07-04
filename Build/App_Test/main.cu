
#include <stdio.h>
#include "Physika_Core/Vectors/vector_fixed.h"
#include "Framework/Base.h"
#include "Framework/Field.h"
#include "Framework/Module.h"
#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include <string>
#include "Physika_Core/Timer/GTimer.h"

int main()
{
	std::shared_ptr<SceneGraph> scene = SceneGraph::getInstance();

	std::shared_ptr<ParticleSystem<DataType3f>> psystem =
		scene->createNewScene<ParticleSystem<DataType3f>>("root");

	psystem->initialize();

	GTimer timer;

	timer.start();
	for (int i = 0; i < 10; i++)
	{
		psystem->advance(0.001f);
	}
	timer.stop();

	std::cout << "Totoal time: " << timer.getEclipsedTime() << std::endl;
	
    return 0;
}