#include <stdio.h>
#include "Physika_Core/Vectors/vector_fixed.h"
#include "Framework/Base.h"
#include "Framework/Field.h"
#include "Framework/Module.h"
#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include <string>
#include "Physika_Core/Timer/GTimer.h"
#include "GlutWindow.h"
#include "GlutMaster.h"
#include "PointRenderer.h"

using namespace Physika;
int main()
{
	std::shared_ptr<SceneGraph> scene = SceneGraph::getInstance();

	std::shared_ptr<ParticleSystem<DataType3f>> psystem =
		scene->createNewScene<ParticleSystem<DataType3f>>("root");

	psystem->initialize();
// 
// 	GTimer timer;
// 	timer.start();
// 	for (int i = 0; i < 10; i++)
// 	{
// 		psystem->advance(0.001f);
// 	}
// 	timer.stop();

	GlutWindow* win = new GlutWindow("first", 1024, 768);

	win->SetSimulation(scene.get());

	CudaPointRender* renderer = new CudaPointRender(psystem->GetNewPositionBuffer()->getDataPtr());
	renderer->SetColorIndex(psystem->GetDensityBuffer()->getDataPtr());
	win->AddRenderer(renderer);

	GlutMaster::instance()->SetActiveWindow("first", win);
	GlutMaster::instance()->SetIdleToCurrentWindow();
	GlutMaster::instance()->EnableIdleFunction();
	GlutMaster::instance()->CallGlutMainLoop();

	return 0;

//	std::cout << "Total time: " << timer.getEclipsedTime() << std::endl;
	
    return 0;
}