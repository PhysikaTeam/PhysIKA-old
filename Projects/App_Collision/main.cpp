#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Log.h"

#include "Rendering/PointRenderModule.h"

#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"
#include "Dynamics/ParticleSystem/Peridynamics.h"

#include "Framework/Collision/CollidableSDF.h"
#include "Framework/Collision/CollidablePoints.h"
#include "Framework/Collision/CollisionSDF.h"
#include "Framework/Framework/Gravity.h"
#include "Dynamics/ParticleSystem/FixedPoints.h"
#include "Framework/Collision/CollisionPoints.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/SolidFluidInteraction.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Rendering/SurfaceMeshRender.h"

using namespace std;
using namespace Physika;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1, 2.0, 1));
	scene.setLowerBound(Vector3f(0, 0.0, 0));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0, 0.0, 0), Vector3f(1, 2.0, 1), 0.015f, true);
	//root->loadSDF("box.sdf", true);

	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
	// 

	root->addChild(sfi);
	sfi->setInteractionDistance(0.02);

	for (int i = 0; i < 6; i++)
	{
		std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
		root->addParticleSystem(bunny);
		if (i % 2 == 0)
		{
			bunny->getSurfaceRender()->setColor(Vector3f(1, 1, 0));
		}
		else
			bunny->getSurfaceRender()->setColor(Vector3f(1, 0, 1));
		
		bunny->setMass(1.0);
		bunny->loadParticles("../Media/bunny/sparse_bunny_points.obj");
		bunny->loadSurface("../Media/bunny/sparse_bunny_mesh.obj");
		bunny->translate(Vector3f(0.4, 0.2 + i * 0.3, 0.8));
		bunny->setVisible(false);
		bunny->getElasticitySolver()->setIterationNumber(10);
		bunny->getElasticitySolver()->setHorizon(0.03);
		bunny->getTopologyMapping()->setSearchingRadius(0.05);

		sfi->addParticleSystem(bunny);
	}

	for (int i = 0; i < 6; i++)
	{
		std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
		root->addParticleSystem(bunny);
		if (i % 2 == 0)
		{
			bunny->getSurfaceRender()->setColor(Vector3f(0.2, 1, 1));
		}
		else
			bunny->getSurfaceRender()->setColor(Vector3f(0.5, 0.3, 1));

		bunny->setMass(1.0);
		bunny->loadParticles("../Media/bunny/sparse_bunny_points.obj");
		bunny->loadSurface("../Media/bunny/sparse_bunny_mesh.obj");
		bunny->translate(Vector3f(0.7, 0.2 + i * 0.3, 0.8));
		bunny->setVisible(false);
		bunny->getElasticitySolver()->setIterationNumber(10);
		bunny->getElasticitySolver()->setHorizon(0.03);
		bunny->getTopologyMapping()->setSearchingRadius(0.05);

		sfi->addParticleSystem(bunny);
	}


// 
// 
// 	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny2 = std::make_shared<ParticleElasticBody<DataType3f>>();
// 	root->addParticleSystem(bunny2);
// 	bunny2->getSurfaceRender()->setColor(Vector3f(1, 0, 1));
// 	bunny2->setMass(1.0);
// 	bunny2->loadParticles("../Media/bunny/sparse_bunny_points.obj");
// 	bunny2->loadSurface("../Media/bunny/sparse_bunny_mesh.obj");
// 	bunny2->translate(Vector3f(0.2, 0.5, 0.5));
// 	bunny2->setVisible(true);
// 	bunny2->getElasticitySolver()->setIterationNumber(10);
// 	bunny2->getElasticitySolver()->setHorizon(0.03);
// 	bunny2->getTopologyMapping()->setSearchingRadius(0.05);
// 
// 	sfi->addParticleSystem(bunny2);
}

int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


