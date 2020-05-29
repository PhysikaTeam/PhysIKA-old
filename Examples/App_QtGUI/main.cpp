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

using namespace std;
using namespace PhysIKA;


int main()
{
	SceneGraph& scene = SceneGraph::getInstance();
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
	bunny->addVisualModule(pRenderer);

	QtApp window;
	window.createWindow(1024, 768);

	window.mainLoop();
    SceneGraph::getInstance().setRootNode(nullptr);
	return 0;
}