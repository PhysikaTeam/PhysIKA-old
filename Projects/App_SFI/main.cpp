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
#include "Rendering/PointRenderModule.h"

using namespace std;
using namespace Physika;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
//	scene.setUpperBound(Vector3f(1, 1.0, 0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.015f, true);

//  
	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(fluid);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(0, 0, 1));
	ptRender->setColorRange(0, 1);
	fluid->addVisualModule(ptRender);

	//fluid->loadParticles("../Media/fluid/fluid_point.obj");
	fluid->loadParticles(Vector3f(0), Vector3f(0.5, 1.0, 1.0), 0.015f);
	fluid->setMass(10);
	//fluid->getVelocity()->connect(fluid->getRenderModule()->m_vecIndex);

	std::shared_ptr<PositionBasedFluidModel<DataType3f>> pbd = std::make_shared<PositionBasedFluidModel<DataType3f>>();
	fluid->getPosition()->connect(pbd->m_position);
	fluid->getVelocity()->connect(pbd->m_velocity);
	fluid->getForce()->connect(pbd->m_forceDensity);
	pbd->setSmoothingLength(0.02);

	fluid->setNumericalModel(pbd);


	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
	// 
	sfi->setInteractionDistance(0.02);
	root->addChild(sfi);

	for (int i = 0; i < 3; i++)
	{
		std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
		root->addParticleSystem(bunny);
		bunny->setMass(1.0);
		bunny->loadParticles("../Media/bunny/sparse_bunny_points.obj");
		bunny->loadSurface("../Media/bunny/sparse_bunny_mesh.obj");
		bunny->translate(Vector3f(0.75, 0.2, 0.4 + i * 0.3));
		bunny->setVisible(false);
		bunny->getElasticitySolver()->setIterationNumber(10);
		bunny->getElasticitySolver()->setHorizon(0.03);
		bunny->getTopologyMapping()->setSearchingRadius(0.05);

		auto sRender = std::make_shared<SurfaceMeshRender>();
		bunny->getSurfaceNode()->addVisualModule(sRender);
		sRender->setColor(Vector3f(i*0.3f, 1 - i*0.3f, 1.0));

		sfi->addParticleSystem(bunny);
	}


	sfi->addParticleSystem(fluid);
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


