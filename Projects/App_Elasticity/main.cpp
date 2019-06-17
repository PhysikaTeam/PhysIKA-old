#include <iostream>
#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"

using namespace Physika;

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);
	bunny->getRenderModule()->setColor(Vector3f(0, 1, 1));
	bunny->setMass(1.0);
	bunny->loadParticles("../Media/bunny/bunny_points.obj");
	bunny->loadSurface("../Media/bunny/bunny_mesh.obj");
	bunny->translate(Vector3f(0.5, 0.2, 0.5));
	bunny->setVisible(true);

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	return 0;
}


