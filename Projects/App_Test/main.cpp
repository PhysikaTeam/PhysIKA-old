#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Physika_GUI/GlutGUI/GLApp.h"

#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Log.h"

#include "Physika_Render/PointRenderModule.h"

#include "Physika_Dynamics/ParticleSystem/PositionBasedFluidModel.h"
#include "Physika_Dynamics/ParticleSystem/Peridynamics.h"

#include "Physika_Framework/Collision/CollidableSDF.h"
#include "Physika_Framework/Collision/CollidablePoints.h"
#include "Physika_Framework/Collision/CollisionSDF.h"
#include "Physika_Framework/Framework/Gravity.h"
#include "Physika_Dynamics/ParticleSystem/FixedPoints.h"
#include "Physika_Framework/Collision/CollisionPoints.h"
#include "Physika_Dynamics/ParticleSystem/VelocityConstraint.h"
#include "Physika_Framework/Framework/DataFlow.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include "Physika_Dynamics/ParticleSystem/ParticleFluid.h"
#include "Physika_Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Physika_Dynamics/RigidBody/RigidBody.h"
#include "Physika_Dynamics/ParticleSystem/StaticBoundary.h"

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

	//std::shared_ptr<ParticleSystem<DataType3f>> root = scene.createNewScene<ParticleSystem<DataType3f>>("root");
	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();

	std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	root->addParticleSystem(child1);
	child1->getRenderModule()->setColor(Vector3f(1, 0, 0));

// 	std::shared_ptr<ParticleSystem<DataType3f>> child2 = std::make_shared<ParticleSystem<DataType3f>>();
// 	root->addParticleSystem(child2);
// 	child2->getRenderModule()->setColor(Vector3f(0, 1, 0));

	std::shared_ptr<ParticleElasticBody<DataType3f>> child3 = std::make_shared<ParticleElasticBody<DataType3f>>();
	root->addParticleSystem(child3);
	child3->getRenderModule()->setColor(Vector3f(0, 0, 1));
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


