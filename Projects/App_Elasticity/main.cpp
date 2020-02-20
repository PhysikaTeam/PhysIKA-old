#include <iostream>
#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"

#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"

using namespace PhysIKA;

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(bunny);

	auto m_pointsRender = std::make_shared<PointRenderModule>();
	m_pointsRender->setColor(Vector3f(0, 1, 1));
	bunny->addVisualModule(m_pointsRender);

	bunny->setMass(1.0);
	bunny->loadParticles("../Media/bunny/bunny_points.obj");
	bunny->loadSurface("../Media/bunny/bunny_mesh.obj");
	bunny->translate(Vector3f(0.5, 0.2, 0.5));
	bunny->setVisible(true);

	auto sRender = std::make_shared<SurfaceMeshRender>();
	bunny->getSurfaceNode()->addVisualModule(sRender);
	sRender->setColor(Vector3f(1, 1, 0));

	bunny->getElasticitySolver()->setIterationNumber(10);

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	return 0;
}


