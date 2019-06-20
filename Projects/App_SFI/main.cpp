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
#include "Rendering/SurfaceMeshRender.h"

using namespace std;
using namespace Physika;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1, 1.0, 0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1, 0.5, 0.3), true);
	//root->loadSDF("box.sdf", true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);
	bunny->getSurfaceRender()->setColor(Vector3f(1, 1, 0));
	bunny->setMass(1.0);
	bunny->loadParticles("../Media/bunny/bunny_points.obj");
	bunny->loadSurface("../Media/bunny/bunny_mesh.obj");
	bunny->translate(Vector3f(0.5, 0.2, 0.5));
	bunny->setVisible(false);
	bunny->getElasticitySolver()->setIterationNumber(10);
// 
// 
	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny2 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny2);
	bunny2->getSurfaceRender()->setColor(Vector3f(1, 0, 1));
	bunny2->setMass(1.0);
	bunny2->loadParticles("../Media/bunny/bunny_points.obj");
	bunny2->loadSurface("../Media/bunny/bunny_mesh.obj");
	bunny2->translate(Vector3f(0.2, 0.2, 0.5));
	bunny2->setVisible(false);
	bunny2->getElasticitySolver()->setIterationNumber(10);
 
	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(fluid);
	fluid->getRenderModule()->setColor(Vector3f(1, 0, 0));
//	fluid->loadParticles("../Media/fluid/fluid_point.obj");
	fluid->setMass(10);
	fluid->getRenderModule()->setColorRange(0, 1);
	fluid->getVelocity()->connect(fluid->getRenderModule()->m_vecIndex);

	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
// 
	root->addChild(sfi);
 	sfi->addParticleSystem(bunny);
 	sfi->addParticleSystem(bunny2);
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


