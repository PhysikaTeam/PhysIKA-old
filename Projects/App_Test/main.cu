#include <iostream>
#include <memory>
#include <cuda.h>
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
#include "Physika_Framework/Framework/DataFlow.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include "Physika_Dynamics/ParticleSystem/ParticleFluid.h"
#include "Physika_Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Physika_Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Physika_Dynamics/RigidBody/RigidBody.h"
#include "Physika_Dynamics/ParticleSystem/StaticBoundary.h"
#include "Physika_Dynamics/ParticleSystem/SolidFluidInteraction.h"
#include "Physika_Core/Algorithm/MatrixFunc.h"
//#include "Physika_Dynamics/ParticleSystem/svd3_cuda2.h"
//#include "svd3_cuda.h"

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

__host__ __device__ __forceinline__
void printMat3(float a11, float a12, float a13,
	float a21, float a22, float a23,
	float a31, float a32, float a33)
{
	printf("%f %f %f \n", a11, a12, a13);
	printf("%f %f %f \n", a21, a22, a23);
	printf("%f %f %f \n", a31, a32, a33);
}

__global__ void svd3_test()
{
	int tid = blockIdx.x;

	float a11, a12, a13, a21, a22, a23, a31, a32, a33;

	//     a11= -0.558253; a12 = -0.0461681; a13 = -0.505735;
	//     a21 = -0.411397; a22 = 0.0365854; a23 = 0.199707;
	//     a31 = 0.285389; a32 =-0.313789; a33 = 0.200189;
// 	a11 = 0.115572; a12 = 0.022244; a13 = 0.188606;
// 	a21 = -0.062891; a22 = 0.012105; a23 = -0.102634;
// 	a31 = 0.120823; a32 = -0.023255; a33 = 0.197176;

	a11 = 1; a12 = 0; a13 = 0;
	a21 = 0; a22 = 0; a23 = 0;
	a31 = 0; a32 = 0; a33 = 0;

	float u11, u12, u13, u21, u22, u23, u31, u32, u33;
	float s11, s12, s13, s21, s22, s23, s31, s32, s33;
	float v11, v12, v13, v21, v22, v23, v31, v32, v33;

	Matrix3f A;
	A(0, 0) = 1; A(0, 1) = 0; A(0, 2) = 0;
	A(1, 0) = 1; A(1, 1) = 1; A(1, 2) = 0;
	A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 1;


	Matrix3f R, U, D, V;
	polarDecomposition(A, R, U, D, V);

	A = U*D*V.transpose();

// 	svd(A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2),
// 		U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2),
// 		D(0, 0), D(1, 1), D(2, 2),
// 		V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2)
// 	svd(a11, a12, a13, a21, a22, a23, a31, a32, a33,
// 		u11, u12, u13, u21, u22, u23, u31, u32, u33,
// 		s11, s22, s33,
// 		v11, v12, v13, v21, v22, v23, v31, v32, v33);

	printMat3(A(0, 0), A(0, 1), A(0, 2), A(1, 0), A(1, 1), A(1, 2), A(2, 0), A(2, 1), A(2, 2));
	printMat3(R(0, 0), R(0, 1), R(0, 2), R(1, 0), R(1, 1), R(1, 2), R(2, 0), R(2, 1), R(2, 2));
	printMat3(U(0, 0), U(0, 1), U(0, 2), U(1, 0), U(1, 1), U(1, 2), U(2, 0), U(2, 1), U(2, 2));
	printMat3(D(0, 0), 0, 0, 0, D(1, 1), 0, 0, 0, D(2, 2));
	printMat3(V(0, 0), V(0, 1), V(0, 2), V(1, 0), V(1, 1), V(1, 2), V(2, 0), V(2, 1), V(2, 2));
}


int main()
{
	svd3_test << <1, 1 >> > (); // 5 blocks, 1 GPU thread each
	cudaDeviceSynchronize();
// 	Matrix3f A;
// 	A(0, 0) = 1; A(0, 1) = 1; A(0, 2) = 0;
// 	A(1, 0) = 1; A(1, 1) = 1; A(1, 2) = 0;
// 	A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 0;

// 	A(0, 0) = 1; A(0, 1) = 0; A(0, 2) = 0;
// 	A(1, 0) = 1; A(1, 1) = 0; A(1, 2) = 0;
// 	A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 0;

// 	A(0, 0) = 0.707107; A(0, 1) = -0.707107; A(0, 2) = 0;
// 	A(1, 0) = 0.707107; A(1, 1) = 0.707107; A(1, 2) = 0;
// 	A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 0;

// 	Matrix3f R, U, D, V;
// 	polarDecomposition(A, R, U, D, V);
// 
// 	A = U*D*V.transpose();

//	polarDecomposition(A, R, U, D);

// 	printf("Mat: \n %f %f %f \n %f %f %f \n %f %f %f \n	R: \n %f %f %f \n %f %f %f \n %f %f %f \n D: \n %f %f %f \n %f %f %f \n %f %f %f \n U :\n %f %f %f \n %f %f %f \n %f %f %f \n V: \n %f %f %f \n %f %f %f \n %f %f %f \n Determinant: %f \n\n",
// 		A(0, 0), A(0, 1), A(0, 2),
// 		A(1, 0), A(1, 1), A(1, 2),
// 		A(2, 0), A(2, 1), A(2, 2),
// 		R(0, 0), R(0, 1), R(0, 2),
// 		R(1, 0), R(1, 1), R(1, 2),
// 		R(2, 0), R(2, 1), R(2, 2),
// 		D(0, 0), D(0, 1), D(0, 2),
// 		D(1, 0), D(1, 1), D(1, 2),
// 		D(2, 0), D(2, 1), D(2, 2),
// 		U(0, 0), U(0, 1), U(0, 2),
// 		U(1, 0), U(1, 1), U(1, 2),
// 		U(2, 0), U(2, 1), U(2, 2),
// 		V(0, 0), V(0, 1), V(0, 2),
// 		V(1, 0), V(1, 1), V(1, 2),
// 		V(2, 0), V(2, 1), V(2, 2),
// 		R.determinant());

// 	Vector3f rotated = R*Vector3f(1, 0, 0);
// 	printf("%f %f %f \n", rotated[0], rotated[1], rotated[2]);

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


