#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/QtGUI/QtApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Rendering/SurfaceMeshRender.h"


using namespace std;
using namespace Physika;


int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);
	bunny->getRenderModule()->setColor(Vector3f(0, 1, 1));
	bunny->setMass(1.0);
	bunny->loadParticles("../Media/bunny/bunny_points.obj");
	bunny->loadSurface("../Media/bunny/bunny_mesh.obj");
	bunny->translate(Vector3f(0.5, 0.2, 0.5));
	bunny->setVisible(true);
	bunny->getSurfaceRender()->setColor(Vector3f(1, 1, 0));
	bunny->getElasticitySolver()->setIterationNumber(10);

	QtApp window;
	window.createWindow(1024, 768);

	window.mainLoop();
    SceneGraph::getInstance().setRootNode(nullptr);
	return 0;
}


