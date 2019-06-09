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
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/SolidFluidInteraction.h"

using namespace std;
using namespace Physika;

//#define DEMO_1
#define DEMO_2
//#define DEMO_3

void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

#ifdef DEMO_1
void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<Node> root = scene.createNewScene<Node>("root");
	auto collidable1 = std::make_shared<CollidableSDF<DataType3f>>();
	root->setCollidableObject(collidable1);

	// 	std::shared_ptr<Node> c1 = root->createChild<Node>("child1");
	// 	auto* pSet1 = new PointSet<Vector3f>();
	// 	c1->setTopologyModule(pSet1);

	//create child node 1
	std::shared_ptr<Node> fluid = root->createChild<Node>("child1");

	fluid->setDt(0.001);

	auto pSet = std::make_shared<PointSet<DataType3f>>();
	fluid->setTopologyModule(pSet);
	std::vector<DataType3f::Coord> positions;
	for (float x = 0.45; x < 0.55; x += 0.005f) {
		for (float y = 0.05; y < 0.15; y += 0.005f) {
			for (float z = 0.45; z < 0.55; z += 0.005f) {
				positions.push_back(DataType3f::Coord(DataType3f::Real(x), DataType3f::Real(y), DataType3f::Real(z)));
			}
		}
	}
	pSet->setPoints(positions);
	int pNum = positions.size();

	positions.clear();
	auto mstate = fluid->getMechanicalState();
	
	auto position = mstate->allocDeviceArray<DataType3f::Coord>(MechanicalState::position(), "Storing the particle positions!");
	auto velocity = mstate->allocDeviceArray<DataType3f::Coord>(MechanicalState::velocity(), "Storing the particle velocities!");
	auto force = mstate->allocDeviceArray<DataType3f::Coord>(MechanicalState::force(), "Storing the force densities!");
	position->setElementCount(pNum);
	velocity->setElementCount(pNum);
	force->setElementCount(pNum);

//	auto restDensity = mstate->createVarField<DataType3f::Real>("rest_density", "reference fluid density", Real(1000));
//	auto position = mstate->createArrayField<DataType3f::Coord, DeviceType::GPU>("fluid_position", "fluid position", pNum);
//	auto velocity = mstate->allocDeviceArray<DataType3f::Coord, DeviceType::GPU>("fluid_velocity", "fluid velocity", pNum);
	// Allocate mechanical states
// 	auto m_mass = HostVarField<DataType3f::Real>::createField(mstate.get(), MechanicalState::mass(), "Particle mass", Real(1));
// 	auto posBuf = DeviceArrayField<DataType3f::Coord>::createField(mstate.get(), MechanicalState::position(), "Particle positions", pNum);
// 	auto velBuf = DeviceArrayField<DataType3f::Coord>::createField(mstate.get(), MechanicalState::velocity(), "Particle velocities", pNum);
// 	auto restPos = DeviceArrayField<DataType3f::Coord>::createField(mstate.get(), MechanicalState::pre_position(), "Old particle positions", pNum);
// 	auto force = DeviceArrayField<DataType3f::Coord>::createField(mstate.get(), MechanicalState::force(), "Particle forces", pNum);
// 	auto restVel = DeviceArrayField<DataType3f::Coord>::createField(mstate.get(), MechanicalState::pre_velocity(), "Particle positions", pNum);
// 	auto rhoBuf = DeviceArrayField<DataType3f::Real>::createField(mstate.get(), MechanicalState::density(), "Particle densities", pNum);
// 	auto adaptNbr = NeighborField<int>::createField(mstate.get(), MechanicalState::particle_neighbors(), "Particle neighbor ids", pNum);
// 	auto normalBuf = DeviceArrayField<DataType3f::Coord>::createField(mstate.get(), MechanicalState::particle_normal(), "Particle normals", pNum);

	auto pS1 = std::make_shared<PositionBasedFluidModel<DataType3f>>();
	//auto pS1 = std::make_shared<Peridynamics<DataType3f>>();
	//	auto pS1 = std::make_shared<ParticleSystem<DataType3f>>();
	fluid->setNumericalModel(pS1);
	position->connect(pS1->m_position);
	velocity->connect(pS1->m_velocity);
	force->connect(pS1->m_forceDensity);

	auto render = std::make_shared<PointRenderModule>();
	render->setColor(Vector3f(0.2f, 0.6, 1.0f));
	//	render->setVisible(false);
	fluid->addVisualModule(render);

	auto cPoints = std::make_shared<CollidablePoints<DataType3f>>();
	fluid->setCollidableObject(cPoints);

// 	auto gravity = std::make_shared<Gravity<DataType3f>>();
// 	gravity->setGravity(Vector3f(0.0, -9.8, 0.0));
// 	fluid->addForceModule(gravity);

	auto cModel = std::make_shared<CollisionSDF<DataType3f>>();
	cModel->setCollidableSDF(collidable1);
	cModel->addCollidableObject(cPoints);
	//	cModel->setSDF(collidable1->getSDF());
	root->addCollisionModel(cModel);
}

#endif

#ifdef DEMO_2
void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0), Vector3f(1), true);
	root->loadCube(Vector3f(0), Vector3f(0.5, 0.5, 1.0), false);
	//root->loadSDF("box.sdf", true);

// 	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
// 	root->addParticleSystem(child1);
// 	child1->getRenderModule()->setColor(Vector3f(1, 0, 0));
// 	child1->scale(0.05);
// 	child1->translate(Vector3f(0.5, 0.3, 0.5));
// 	child1->setMass(100);
// 	child1->getRenderModule()->setColorRange(0, 2);

//	child1->loadParticles("D:/OpenSource/SDFGen/bowl.obj");

// 	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
// 	root->addRigidBody(rigidbody);
// 	rigidbody->loadShape("D:/OpenSource/SDFGen/bowl.obj");

	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
	root->addParticleSystem(child3);
	child3->getRenderModule()->setColor(Vector3f(0, 0, 1));
	child3->setMass(1.0);
//  	child3->loadParticles("D:/OpenSource/SDFGen/bowl.obj");
//  	child3->loadSurface("D:/OpenSource/SDFGen/bowl.obj");
	child3->scale(0.05);
	child3->translate(Vector3f(0.5, 0.6, 0.5));

// 	std::shared_ptr<ParticleElasticBody<DataType3f>> child3 = std::make_shared<ParticleElasticBody<DataType3f>>();
// 	root->addParticleSystem(child3);
// 	child3->getRenderModule()->setColor(Vector3f(0, 0, 1));
// 	child3->setVisible(false);
// 	child3->setMass(1.0);
// 	child3->loadParticles("D:/s1_asm_points.obj");
// 	child3->loadSurface("D:/s1_asm2.obj");
// 	child3->scale(0.00005);
// 	child3->translate(Vector3f(0.4, 0.6, 0.5));

// 	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child4 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
// 	root->addParticleSystem(child4);
// 	child4->getRenderModule()->setColor(Vector3f(0, 0, 1));
// 	child4->setMass(1.0);
// 	//  	child3->loadParticles("D:/OpenSource/SDFGen/bowl.obj");
// 	//  	child3->loadSurface("D:/OpenSource/SDFGen/bowl.obj");
// 	child4->scale(0.05);
// 	child4->translate(Vector3f(0.35, 0.2, 0.5));


// 	std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
// 	root->addParticleSystem(child3);
// 	child3->getRenderModule()->setColor(Vector3f(0, 0, 1));
// 	child3->setMass(1.0);
// 	// 	child3->loadParticles("D:/OpenSource/SDFGen/bowl.obj");
// 	// 	child3->loadSurface("D:/OpenSource/SDFGen/bowl.obj");
// 	child3->scale(0.05);
// 	child3->translate(Vector3f(0.2, 0.3, 0.5));
 
  	std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
  
	root->addChild(sfi);

//	sfi->addParticleSystem(child1);
	sfi->addParticleSystem(child3);
//	sfi->addParticleSystem(child4);
}
#endif

#ifdef DEMO_3
void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	std::shared_ptr<RigidBody<DataType3f>> root = scene.createNewScene<RigidBody<DataType3f>>("root");
}
#endif


int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
	return 0;
}


