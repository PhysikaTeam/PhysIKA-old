#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "GUI/QtGUI/QtApp.h"
#include "GUI/QtGUI/PVTKSurfaceMeshRender.h"
#include "GUI/QtGUI/PVTKPointSetRender.h"

#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Framework/ControllerAnimation.h"

#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/ParticleEmitter.h"
#include "Dynamics/ParticleSystem/ParticleEmitterRound.h"

using namespace std;
using namespace PhysIKA;


int main()
{
/*	SceneGraph& scene = SceneGraph::getInstance();
	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);
	root->setName("Root");

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);
//	bunny->getRenderModule()->setColor(Vector3f(0, 1, 1));
	bunny->setMass(1.0);
	bunny->loadParticles("../../Media/bunny/bunny_points.obj");
	bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
	bunny->translate(Vector3f(0.5, 0.2, 0.5));
	bunny->setVisible(true);
//	bunny->getSurfaceRender()->setColor(Vector3f(1, 1, 0));
	bunny->getElasticitySolver()->setIterationNumber(10);

	auto renderer = std::make_shared<PVTKSurfaceMeshRender>();
	renderer->setName("VTK Mesh Renderer");
	bunny->getSurfaceNode()->addVisualModule(renderer);


	auto pRenderer = std::make_shared<PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");
	bunny->addVisualModule(pRenderer);*/


	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.5, 1, 1.5));
	scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(-0.5, 0, -0.5), Vector3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../Media/bowl/bowl.sdf", false);
	root->setName("static");

	

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);
	child1->setName("fluid");

	std::shared_ptr<ParticleEmitterRound<DataType3f>> child2 = std::make_shared<ParticleEmitterRound<DataType3f>>();
	root->addParticleSystem(child2);
	//child1->loadParticles("../Media/fluid/fluid_point.obj");
	//child1->loadParticles(Vector3f(0.5, 0.2, 0.4), Vector3f(0.7, 1.5, 0.6), 0.005);
	child1->setMass(100);

	

	child2->currentForce()->connect(child1->currentForce());
	child2->currentPosition()->connect(child1->currentPosition());
	child2->currentVelocity()->connect(child1->currentVelocity());

	child2->setInfo(Vector3f(0.5,0.5,0.5), Vector3f(0.0, 0.0, 0.0), 0.1, 0.005);

	auto pRenderer = std::make_shared<PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");
	child1->addVisualModule(pRenderer);




	QtApp window;
	window.createWindow(1024, 768);

	auto classMap = Object::getClassMap();

	for (auto const c : *classMap)
		std::cout << "Class Name: " << c.first << std::endl;

	window.mainLoop();

	return 0;
}