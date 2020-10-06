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
using namespace PhysIKA;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setFrameRate(500);
	scene.setUpperBound(Vector3f(1, 2.0, 1));
	scene.setLowerBound(Vector3f(0, 0.0, 0));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0, 0.0, 0), Vector3f(1, 2.0, 1), 0.015f, true);
	//root->loadSDF("box.sdf", true);

	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
	// 

	root->addChild(sfi);
	sfi->setInteractionDistance(0.03);

	for (int i = 0; i < 6; i++)
	{
		std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
		root->addParticleSystem(bunny);

		auto sRender = std::make_shared<SurfaceMeshRender>();
		bunny->getSurfaceNode()->addVisualModule(sRender);
		
		if (i % 2 == 0)
		{
			sRender->setColor(Vector3f(1, 1, 0));
		}
		else
			sRender->setColor(Vector3f(1, 0, 1));
		
		bunny->varHorizon()->setValue(0.03f);
		bunny->setMass(1.0);
		bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
		bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
		bunny->translate(Vector3f(0.4, 0.2 + i * 0.3, 0.8));
		bunny->setVisible(false);
		bunny->getElasticitySolver()->setIterationNumber(10);
		//bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
		bunny->getTopologyMapping()->setSearchingRadius(0.05);

		sfi->addParticleSystem(bunny);
	}
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


