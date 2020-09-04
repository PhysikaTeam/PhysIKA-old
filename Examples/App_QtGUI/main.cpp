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

using namespace std;
using namespace PhysIKA;


int main()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.5, 1.5, 1.5));
	scene.setLowerBound(Vector3f(-1.5, -0.5, -1.5));



	std::shared_ptr<MeshBoundary<DataType3f>> root = scene.createNewScene<MeshBoundary<DataType3f>>();
	root->loadMesh("../../Media/bowl/b3.obj");
	root->setName("StaticMesh");
	//root->loadMesh();
	

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	//root->addParticleSystem(child1);
	child1->setName("fluid");
//	child1->setBlockCoord(400, 50);

//	std::shared_ptr<ParticleEmitterSquare<DataType3f>> child2 = std::make_shared<ParticleEmitterSquare<DataType3f>>();
	root->addParticleSystem(child1);
	//child2->setActive(false);

//	child2->addOutput(child1,child2);
	//child1->addChild(child2);
//	child1->addParticleEmitter(child2);

	//child1->loadParticles("../Media/fluid/fluid_point.obj");
//	child1->loadParticles(Vector3f(-0.5, 0.005, -0.5), Vector3f(-0.4, 0.4, 0.5), 0.005);
// 	child1->setMass(100);
// 
// 	
// 	//child2->setInfo(Vector3f(0.3,0.5,0.3), Vector3f(1.0, 1.0, 1.0), 0.1, 0.005);
// 	child2->setInfo(Vector3f(0.0, 0.5, 0.0), Vector3f(-1.0, -1.0, -1.0), 0.1, 0.005);

// 	auto pRenderer = std::make_shared<PVTKPointSetRender>();
// 	pRenderer->setName("VTK Point Renderer");
// 	child1->addVisualModule(pRenderer);

	printf("outside 1\n");
	
// 	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
// 	root->addRigidBody(rigidbody);
// 	rigidbody->loadShape("../../Media/bowl/b3.obj");
// 	printf("outside 2\n");
// 	auto sRenderer = std::make_shared<PVTKSurfaceMeshRender>();
// 	sRenderer->setName("VTK Surface Renderer");
// 	rigidbody->getSurface()->addVisualModule(sRenderer);
// 	rigidbody->setActive(false);
	
	
	
	printf("outside 3\n");
	QtApp window;
	window.createWindow(1024, 768);
	printf("outside 4\n");
	auto classMap = Object::getClassMap();

	for (auto const c : *classMap)
		std::cout << "Class Name: " << c.first << std::endl;

	window.mainLoop();

	return 0;
}