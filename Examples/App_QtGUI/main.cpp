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
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleEmitterSquare.h"

#include "Dynamics/ParticleSystem/MeshBoundary.h"

#include "Dynamics/ParticleSystem/ParticleWriter.h"

#include "Dynamics/ParticleSystem/MeshBoundary.h"
#include "Framework/Topology/TriangleSet.h"

#include "Dynamics/ParticleSystem/SemiAnalyticalSFINode.h"
#include "Dynamics/ParticleSystem/TriangularSurfaceMeshNode.h"

using namespace std;
using namespace PhysIKA;


void creare_scene_init()
{
	float* a = nullptr;
	float& b = *a;

	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.5, 1.5, 1.5));
	scene.setLowerBound(Vector3f(-1.5, -0.5, -1.5));



	std::shared_ptr<MeshBoundary<DataType3f>> root = scene.createNewScene<MeshBoundary<DataType3f>>();
	root->loadMesh("../../Media/bowl/b3.obj");
	root->setName("StaticMesh");
	//root->loadMesh();
	

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);
	child1->setName("fluid");
//	root->addParticleSystem(child1);
//	child1->setBlockCoord(400, 50);

 	std::shared_ptr<ParticleEmitterSquare<DataType3f>> child2 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
// 	root->addParticleSystem(child1);
	//child2->setActive(false);

//	child2->addOutput(child1,child2);
//	child1->addChild(child2);
	child1->addParticleEmitter(child2);

	//child1->loadParticles("../Media/fluid/fluid_point.obj");
//	child1->loadParticles(Vector3f(0, 0.5, -0.5), Vector3f(0.01, 0.51, 0.5), 0.005);
	child1->setMass(100);

	
// 	//child2->setInfo(Vector3f(0.3,0.5,0.3), Vector3f(1.0, 1.0, 1.0), 0.1, 0.005);
 //	child2->setInfo(Vector3f(0.0, 0.5, 0.0), Vector3f(-1.0, -1.0, -1.0), 0.1, 0.005);
// 
	auto pRenderer = std::make_shared<PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");
	child1->addVisualModule(pRenderer);
    printf("outside visual\n");
// 	printf("outside 1\n");
// 	
	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../../Media/bowl/b3.obj");
	printf("outside 2\n");
	auto sRenderer = std::make_shared<PVTKSurfaceMeshRender>();
	sRenderer->setName("VTK Surface Renderer");
	rigidbody->getSurface()->addVisualModule(sRenderer);
	rigidbody->setActive(false);
	
	
	
	printf("outside 3\n");
	QtApp window;
	window.createWindow(1024, 768);
	printf("outside 4\n");
	auto classMap = Object::getClassMap();

	for (auto const c : *classMap)
		std::cout << "Class Name: " << c.first << std::endl;

	window.mainLoop();

	//return 0;
}

void create_scene_semianylitical()
{

	float* a = nullptr;
	float& b = *a;

	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.2));
	scene.setLowerBound(Vector3f(-0.2));

	scene.setFrameRate(1000);

	std::shared_ptr<SemiAnalyticalSFINode<DataType3f>> root = scene.createNewScene<SemiAnalyticalSFINode<DataType3f>>();
	root->setName("SemiAnalyticalSFI");
	//root->loadMesh();

	auto writer = std::make_shared<ParticleWriter<DataType3f>>();
	writer->setNamePrefix("particles_");
	root->getParticlePosition()->connect(&writer->m_position);
	root->getParticleMass()->connect(&writer->m_color_mapping);
	//root->addModule(writer);


	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);
	child1->setName("fluid");
	//child1->loadParticles(Vector3f(0.75, 0.05, 0.75), Vector3f(0.85, 0.35, 0.85), 0.005);
	child1->setMass(1);

	std::shared_ptr<ParticleEmitterSquare<DataType3f>> child3 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
	child1->addParticleEmitter(child3);


	auto pRenderer = std::make_shared<PVTKPointSetRender>();
	pRenderer->setName("VTK Point Renderer");


	//auto pRenderer = std::make_shared<PointRenderModule>();
	//pRenderer->setColor(Vector3f(0, 0, 1));

	child1->addVisualModule(pRenderer);


	std::shared_ptr <TriangularSurfaceMeshNode<DataType3f>> child2 = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("boundary");
	child2->getTriangleSet()->loadObjFile("../../Media/standard/standard_cube_01.obj");

	root->addTriangularSurfaceMeshNode(child2);


	QtApp window;
	window.createWindow(1024, 768);
	//printf("outside 4\n");
	auto classMap = Object::getClassMap();

	for (auto const c : *classMap)
		std::cout << "Class Name: " << c.first << std::endl;

	window.mainLoop();



}

int main()
{
	//creare_scene_init();
	create_scene_semianylitical();
	return 0;
}