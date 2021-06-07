#include "demoPBD.h"
#include "GUI/GlutGUI/GLApp.h"
//#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolverNode.h"
//#include "Dynamics/RigidBody/PBDRigid/PBDRigidSysNode.h"
#include "Dynamics/RigidBody/PBDRigid/HeightFieldPBDInteractionNode.h"
#include "Dynamics/RigidBody/RigidUtil.h"

#include "Rendering/RigidMeshRender.h"

#include "TestRigidUtil.h"

#include "Dynamics/HeightField/HeightFieldMesh.h"
#include <random>


using namespace PhysIKA;

void DemoPBDPositionConstraint::run()
{
	SceneGraph& scene = SceneGraph::getInstance();
	std::shared_ptr<PBDSolverNode> root = scene.createNewScene<PBDSolverNode>();
	root->setDt(0.016);
	//root->setUseGPU(useGPU);
	root->getSolver()->setUseGPU(useGPU);

	std::vector<std::pair<RigidBody2_ptr, int>> rigids(N + 1);
	//std::shared_ptr<SphericalJoint> joint[N];
	std::vector<PBDJoint<double>> joints;
	joints.resize(N);

	//auto prigid0 = std::make_shared<RigidBody2<DataType3f>>("rigid");
	//int id0 = root->addRigid(prigid0);
	//rigids[0] = std::make_pair(prigid0, id0);
	rigids[0] = std::make_pair(RigidBody2_ptr(), -1);

	for (int i = 0; i < N; ++i)
	{
		/// rigids body
		auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
		int id = root->addRigid(prigid);
		rigids[i + 1] = std::make_pair(prigid, id);


		auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
		renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
		prigid->addVisualModule(renderModule);

		prigid->loadShape("../../Media/standard/standard_cube.obj");
		prigid->setExternalForce(Vector3f(0, -9, 0));
	}

	std::default_random_engine e(time(0));
	std::uniform_real_distribution<float> u(-2.0, 2.0);
	std::uniform_real_distribution<float> u1(-1, 1);

	float box_sx = 0.10, box_sy = 0.30, box_sz = 0.10;
	//box_sx *= 0.1; box_sy *= 0.1; box_sz *= 0.1;
	float rigid_mass = 12;
	float ixy = 0.0, ixz = 0.0, iyz = 0.0;
	float ixx = (box_sy*box_sy + box_sz * box_sz);
	float iyy = (box_sx*box_sx + box_sz * box_sz);
	float izz = (box_sx*box_sx + box_sy * box_sy);

	//Vector3f rigid_r0(0, box_sy * N + box_sy / 2.0, 0);
	//Quaternionf rigid_q0;// = Quaternionf(0, 0, 1, 1).normalize();
	//prigid0->setGlobalR(rigid_r0);
	//prigid0->setI(Inertia<float>(0, Vector3f()));

	for (int i = 0; i < N; ++i)
	{

		Vector3d joint_r0(0, -box_sy / 2.0, 0);
		if (!rigids[i].first)
		{
			joint_r0 = Vector3d(0, box_sy * N, 0);
		}
		Vector3d joint_r1(0, box_sy / 2.0, 0);
		Quaternion<double> joint_q0;
		Quaternion<double> joint_q1;
		joints[i].localPose0.position = joint_r0;
		joints[i].localPose0.rotation = joint_q0;
		joints[i].localPose1.position = joint_r1;
		joints[i].localPose1.rotation = joint_q1;

		joints[i].compliance = 0.000000;
		joints[i].rotationXLimited = false;// i == 0 ? false : true;
		//joints[i].minAngleX = -0.01; joints[i].maxAngleX = 0.01;
		joints[i].rotationYLimited = false;
		joints[i].minAngleY = -0.5; joints[i].maxAngleY = 0.5;
		joints[i].rotationZLimited = false;

		root->addPBDJoint(joints[i], rigids[i].second, rigids[i + 1].second);

		/// ********** rigids 1
		Quaternion<float> rigid_q;
		//if(i<2)
			rigid_q = Quaternionf(0, 0, 0.5, 1).normalize();

		//Vector3f rigid_r(0, box_sy * (N - i) - box_sy / 2.0, 0);
		Vector3f rigid_r;
		if (rigids[i].first)
		{
			rigid_r = rigids[i].first->getGlobalR()
				+ rigids[i].first->getGlobalQ().rotate(Vector3f(joint_r0[0], joint_r0[1], joint_r0[2]));
		}
		else
		{
			rigid_r = Vector3f(joint_r0[0], joint_r0[1], joint_r0[2]);
		}
		rigid_r -= rigid_q.rotate(Vector3f(joint_r1[0], joint_r1[1], joint_r1[2]));

		rigids[i + 1].first->setGlobalR(rigid_r);
		rigids[i + 1].first->setGlobalQ(rigid_q);
		rigids[i + 1].first->setGeometrySize(box_sx, box_sy, box_sz);
		rigids[i + 1].first->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

		//if (i != 0)
		//{
		//	rigids[i + 1].first->setLinearVelocity(Vector3f(0.0, 0.0, 0.5));
		//}

		//if (false)
		//{
		//	/// set velocity
		//	SpatialVector<float> relv(u(e), u(e), u(e), 0, 0, 0);												///< relative velocity, in successor frame.
		//	joint[i]->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[idx_map[id]]));				///< map relative velocity into joint space vector.
		//	motion_state->m_v[id] = (joint[i]->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id]])));	///< set 
		//}
	}



	GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
}

void DemoPBDRotationConstraint::run()
{
	SceneGraph& scene = SceneGraph::getInstance();
	std::shared_ptr<PBDSolverNode> root = scene.createNewScene<PBDSolverNode>();
	root->setDt(0.016);
	//root->setUseGPU(useGPU);
	root->getSolver()->setUseGPU(useGPU);


	std::vector<std::pair<RigidBody2_ptr, int>> rigids(N + 1);
	//std::shared_ptr<SphericalJoint> joint[N];
	std::vector<PBDJoint<double>> joints;
	joints.resize(N);

	//auto prigid0 = std::make_shared<RigidBody2<DataType3f>>("rigid");
	//int id0 = root->addRigid(prigid0);
	//rigids[0] = std::make_pair(prigid0, id0);
	rigids[0] = std::make_pair(RigidBody2_ptr(), -1);

	for (int i = 0; i < N; ++i)
	{
		/// rigids body
		auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
		int id = root->addRigid(prigid);
		rigids[i + 1] = std::make_pair(prigid, id);

		auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
		renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
		prigid->addVisualModule(renderModule);

		prigid->loadShape("../../Media/standard/standard_cube.obj");
		prigid->setExternalForce(Vector3f(0, -9, 0));
	}

	std::default_random_engine e(time(0));
	std::uniform_real_distribution<float> u(-2.0, 2.0);
	std::uniform_real_distribution<float> u1(-1, 1);

	float box_sx = 0.10, box_sy = 0.30, box_sz = 0.10;
	//box_sx *= 0.1; box_sy *= 0.1; box_sz *= 0.1;
	float rigid_mass = 12;
	float ixy = 0.0, ixz = 0.0, iyz = 0.0;
	float ixx = (box_sy*box_sy + box_sz * box_sz);
	float iyy = (box_sx*box_sx + box_sz * box_sz);
	float izz = (box_sx*box_sx + box_sy * box_sy);

	//Vector3f rigid_r0(0, box_sy * N + box_sy / 2.0, 0);
	//Quaternionf rigid_q0;// = Quaternionf(0, 0, 1, 1).normalize();
	//prigid0->setGlobalR(rigid_r0);
	//prigid0->setI(Inertia<float>(0, Vector3f()));

	for (int i = 0; i < N; ++i)
	{

		Vector3d joint_r0(0, -box_sy / 2.0, 0);
		if (!rigids[i].first)
		{
			joint_r0 = Vector3d(0, box_sy * N, 0);
		}
		Vector3d joint_r1(0, box_sy / 2.0, 0);
		Quaternion<double> joint_q0;
		Quaternion<double> joint_q1;
		joints[i].localPose0.position = joint_r0;
		joints[i].localPose0.rotation = joint_q0;
		joints[i].localPose1.position = joint_r1;
		joints[i].localPose1.rotation = joint_q1;

		joints[i].compliance = 0.000000;
		joints[i].angDamping = 10000;
		joints[i].linDamping = 10000;

		joints[i].rotationXLimited = true;// i == 0 ? false : true;
		//joints[i].minAngleX = -0.01; joints[i].maxAngleX = 0.01;
		joints[i].rotationYLimited = true;
		joints[i].minAngleY = -0.5; joints[i].maxAngleY = 0.5;
		joints[i].rotationZLimited = false;

		root->addPBDJoint(joints[i], rigids[i].second, rigids[i + 1].second);

		/// ********** rigids 1
		Quaternion<float> rigid_q;
		//if(i==0)
		//rigid_q = Quaternionf(0, 0, 0.5, 1).normalize();

		//Vector3f rigid_r(0, box_sy * (N - i) - box_sy / 2.0, 0);
		Vector3f rigid_r;
		if (rigids[i].first)
		{
			rigid_r = rigids[i].first->getGlobalR()
				+ rigids[i].first->getGlobalQ().rotate(Vector3f(joint_r0[0], joint_r0[1], joint_r0[2]));
		}
		else
		{
			rigid_r = Vector3f(joint_r0[0], joint_r0[1], joint_r0[2]);
		}
		rigid_r -= rigid_q.rotate(Vector3f(joint_r1[0], joint_r1[1], joint_r1[2]));

		rigids[i + 1].first->setGlobalR(rigid_r);
		rigids[i + 1].first->setGlobalQ(rigid_q);
		rigids[i + 1].first->setGeometrySize(box_sx, box_sy, box_sz);
		rigids[i + 1].first->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

		//if (i != 0)
		{
			rigids[i + 1].first->setLinearVelocity(Vector3f(0.0, 0.0, 4.5));
		}

		//if (false)
		//{
		//	/// set velocity
		//	SpatialVector<float> relv(u(e), u(e), u(e), 0, 0, 0);												///< relative velocity, in successor frame.
		//	joint[i]->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[idx_map[id]]));				///< map relative velocity into joint space vector.
		//	motion_state->m_v[id] = (joint[i]->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id]])));	///< set 
		//}
	}



	GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
}


//
//void DemoPBDCommonRigid::run()
//{
//	if (N <= 0)
//		N = 1;
//
//	SceneGraph& scene = SceneGraph::getInstance();
//	std::shared_ptr<PBDRigidSysNode> root = scene.createNewScene<PhysIKA::PBDRigidSysNode>();
//	root->setDt(0.016);
//	//root->setUseGPU(useGPU);
//	root->getSolver()->setUseGPU(useGPU);
//
//
//	std::vector<RigidBody2_ptr> rigids(N + 1);
//	//std::vector<PBDJoint<double>> joints;
//	//joints.resize(N);
//
//	float box_sx = 0.20, box_sy = 0.20, box_sz = 0.20;
//	//box_sx *= 0.1; box_sy *= 0.1; box_sz *= 0.1;
//	float rigid_mass = 12;
//	float ixy = 0.0, ixz = 0.0, iyz = 0.0;
//	float ixx = (box_sy*box_sy + box_sz * box_sz);
//	float iyy = (box_sx*box_sx + box_sz * box_sz);
//	float izz = (box_sx*box_sx + box_sy * box_sy);
//
//
//	for (int i = 0; i < N; ++i)
//	{
//		/// rigids body
//		auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
//		int id = root->addRigid(prigid);
//		rigids[i] = prigid;
//
//		auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
//		renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
//		prigid->addVisualModule(renderModule);
//
//		prigid->loadShape("../../Media/standard/standard_cube.obj");
//		prigid->setExternalForce(Vector3f(0, -9, 0));
//
//		if (i == 0)
//		{
//			prigid->setGlobalR(Vector3f(0, -0.1, 0));
//			prigid->setGlobalQ(Quaternionf(0,0,0,1));
//			prigid->setGeometrySize(10, 0.2, 10);
//			prigid->setI(Inertia<float>(0, Vector3f(0, 0, 0)));
//		}
//		else
//		{
//			Quaternion<float> rigid_q(0.3, 0, 0.5, 0.5); rigid_q.normalize();
//			Vector3f rigid_r;
//			rigid_r[1] = 1.5 *box_sy * i;
//			rigid_r[0] = 0.9 *box_sy * i;
//
//			prigid->setGlobalR(rigid_r);
//			prigid->setGlobalQ(rigid_q);
//			prigid->setGeometrySize(box_sx, box_sy, box_sz);
//			prigid->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));
//		}
//	}
//
//	std::default_random_engine e(time(0));
//	std::uniform_real_distribution<float> u(-2.0, 2.0);
//	std::uniform_real_distribution<float> u1(-1, 1);
//
//
//
//	GLApp window;
//	window.createWindow(1024, 768);
//	window.mainLoop();
//}


DemoPBDSingleHFCollide* DemoPBDSingleHFCollide::m_instance = 0;
void DemoPBDSingleHFCollide::createScene()
{
	int N = 1;

	SceneGraph& scene = SceneGraph::getInstance();
	std::shared_ptr<HeightFieldPBDInteractionNode> root = scene.createNewScene<HeightFieldPBDInteractionNode>();
	root->setDt(0.016);
	root->getSolver()->setUseGPU(true);
	int nx = 128, ny = 128;
	float dl = 0.01;

	//m_groundRigidInteractor = std::make_shared<HeightFieldPBDInteractionNode>();
	//m_groundRigidInteractor->setRigidBodySystem(m_car->m_rigidSystem);
	root->setSize(nx, ny, dl, dl);
	//m_groundRigidInteractor->setTerrainInfo(terraininfo);

	Array2D<double, DeviceType::CPU> height;
	height.resize(nx, ny);
	memset(height.GetDataPtr(), 0, sizeof(float) * nx * ny);

	DeviceHeightField1d& terrain = root->getHeightField();
	Array2D < double , DeviceType::GPU > * terrain_ = &terrain;
	Function1Pt::copy(*terrain_, height);
	root->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);
	//m_groundRigidInteractor->setDetectionMethod(HeightFieldTerrainRigidInteractionNode::HFDETECTION::FACEVISE);


	Vector3f box_size(0.1, 0.1, 0.1);
	//box_sx *= 0.1; box_sy *= 0.1; box_sz *= 0.1;
	float rigid_mass = 12;
	Vector3f rigid_inertia = RigidUtil::calculateCubeLocalInertia(rigid_mass, box_size);


	//float ixy = 0.0, ixz = 0.0, iyz = 0.0;
	//float ixx = (box_sy*box_sy + box_sz * box_sz);
	//float iyy = (box_sx*box_sx + box_sz * box_sz);
	//float izz = (box_sx*box_sx + box_sy * box_sy);


	for (int i = 0; i < N; ++i)
	{
		/// rigids body
		auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
		int id = root->addRigid(prigid);
		prigid->setDt(0.016);
		//rigids[i] = prigid;

		auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
		renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
		prigid->addVisualModule(renderModule);

		prigid->loadShape("../../Media/standard/standard_cube.obj");
		(std::dynamic_pointer_cast<TriangleSet<DataType3f>>)(prigid->getTopologyModule())->scale(box_size);
		prigid->setExternalForce(Vector3f(0, -9 * rigid_mass, 0));

		if (i == 0)
		{
			prigid->setGlobalR(Vector3f(0, 0.5, 0));
			prigid->setGlobalQ(Quaternionf(0, 0, 0, 1));
			prigid->setI(Inertia<float>(rigid_mass, rigid_inertia));
		}
		else
		{
			Quaternion<float> rigid_q(0.3, 0, 0.5, 0.5); rigid_q.normalize();
			Vector3f rigid_r;
			rigid_r[1] = 1.5 *box_size[1] * i;
			rigid_r[0] = 0.9 *box_size[1] * i;

			prigid->setGlobalR(rigid_r);
			prigid->setGlobalQ(rigid_q);
			prigid->setI(Inertia<float>(rigid_mass, rigid_inertia));
		}
	}



	//// Translate camera position
	//auto& camera_ = this->activeCamera();
	////camera_.translate(Vector3f(0, 1.5, 3));
	////camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
	//Vector3f camPos(0, 1.5, 5);
	//camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}



DemoCollisionTest* DemoCollisionTest::m_instance = 0;
void DemoCollisionTest::build(bool useGPU)
{
	int ny = 1057, nx = 1057;
	float hScale = 0.1;
	float dl = 0.22 * hScale;
	float hOffset = -28.5;


	SceneGraph& scene = SceneGraph::getInstance();

	m_groundRigidInteractor = scene.createNewScene<HeightFieldPBDInteractionNode>();
	m_groundRigidInteractor->setSize(nx, ny, dl, dl);
	m_groundRigidInteractor->getSolver()->m_numSubstep = 5;
	m_groundRigidInteractor->getSolver()->m_numContactSolveIter = 30;
	m_groundRigidInteractor->getSolver()->setUseGPU(useGPU);
	m_groundRigidInteractor->setDt(0.016);


	// Energy computation module.
	std::shared_ptr<RigidEnergyComputeModule> engComputer = std::make_shared<RigidEnergyComputeModule>();
	engComputer->outfile = std::string("D:\\Projects\\physiKA\\PostProcess\\DemoCollisionTest_eng") + 
		(useGPU? std::string("_GPU"): std::string("_CPU")) + ".txt";
	engComputer->maxStep = 1000;
	m_groundRigidInteractor->addCustomModule(engComputer);




	// Land height field
	HostHeightField1d height;
	height.resize(nx, ny);
	memset(height.GetDataPtr(), 0, sizeof(double) * nx * ny);

	// Set node terrain.
	DeviceHeightField1d& terrain = m_groundRigidInteractor->getHeightField();
	DeviceHeightField1d* terrain_ = &terrain;
	Function1Pt::copy(*terrain_, height);
	m_groundRigidInteractor->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);

	// Land rigid.
	{
		auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
		m_groundRigidInteractor->addChild(landrigid);

		// Mesh triangles.
		auto triset = std::make_shared< TriangleSet<DataType3f>>();
		landrigid->setTopologyModule(triset);

		// Generate mesh.
		auto& hfland = terrain;
		HeightFieldMesh hfmesh;
		hfmesh.generate(triset, hfland);

		// Mesh renderer.
		auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
		renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
		landrigid->addVisualModule(renderModule);
	}



	Vector3f box_size(0.1, 0.1, 0.1);
	float rigid_mass = 12;
	Vector3f rigid_inertia = RigidUtil::calculateCubeLocalInertia(rigid_mass, box_size);


	/// rigids body
	auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
	int id = m_groundRigidInteractor->addRigid(prigid);
	prigid->setDt(0.016);
	//rigids[i] = prigid;

	auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
	renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
	prigid->addVisualModule(renderModule);

	prigid->loadShape("../../Media/standard/standard_cube.obj");
	(std::dynamic_pointer_cast<TriangleSet<DataType3f>>)(prigid->getTopologyModule())->scale(box_size);
	prigid->setExternalForce(Vector3f(0, -9 * rigid_mass, 0));
	prigid->setGlobalR(Vector3f(0, 0.5, 0));
	prigid->setGlobalQ(Quaternionf(0, 0, 0, 1));
	prigid->setI(Inertia<float>(rigid_mass, rigid_inertia));
	prigid->setLinearDamping(0.0);
	prigid->setAngularDamping(0.0);

	engComputer->allRigids.push_back(prigid);



	this->createWindow(1024, 768);
	this->mainLoop();
}


DemoPendulumTest* DemoPendulumTest::m_instance = 0;
void DemoPendulumTest::build(bool useGPU)
{
	int N = 50;
	int M = 5;
	double dt = 0.016;

	SceneGraph& scene = SceneGraph::getInstance();
	std::shared_ptr<PBDSolverNode> root = scene.createNewScene<PBDSolverNode>();
	root->setDt(dt);
	//root->setUseGPU(useGPU);
	root->getSolver()->setUseGPU(useGPU);
	root->getSolver()->m_numSubstep = 50;

	Vector3f box_size(0.05, 0.15, 0.05);
	float rigid_mass = 1;
	Vector3f rigid_inertia = RigidUtil::calculateCubeLocalInertia(rigid_mass, box_size);

	for (int mi = 0; mi < M; ++mi)
	{
		for (int mj = 0; mj < M; ++mj)
		{
			std::vector<std::pair<RigidBody2_ptr, int>> rigids(N + 1);
			std::vector<PBDJoint<double>> joints;
			joints.resize(N);

			rigids[0] = std::make_pair(RigidBody2_ptr(), -1);

			

			for (int i = 0; i < N; ++i)
			{
				/// rigids body
				auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
				int id = root->addRigid(prigid);
				prigid->setDt(dt);
				prigid->setI(Inertia<float>(rigid_mass, rigid_inertia));

				rigids[i + 1] = std::make_pair(prigid, id);


				auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
				renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
				prigid->addVisualModule(renderModule);

				prigid->loadShape("../../Media/standard/standard_cube.obj");
				(std::dynamic_pointer_cast<TriangleSet<DataType3f>>)(prigid->getTopologyModule())->scale(box_size / 2.0);
				prigid->setExternalForce(Vector3f(0, -9 * rigid_mass, 0));

				prigid->setLinearDamping(0.);
				prigid->setAngularDamping(0.);
			}

			std::default_random_engine e(time(0));
			std::uniform_real_distribution<float> u(-2.0, 2.0);
			std::uniform_real_distribution<float> u1(-1, 1);

			for (int i = 0; i < N; ++i)
			{

				Vector3d joint_r0(0, -box_size[1] / 2.0, 0);
				if (!rigids[i].first)
				{
					joint_r0 = Vector3d((mi - M / 2.0)*0.5, box_size[1] * N, (mj - M / 2.0)*0.5);
				}
				Vector3d joint_r1(0, box_size[1] / 2.0, 0);
				Quaternion<double> joint_q0;
				Quaternion<double> joint_q1;
				joints[i].localPose0.position = joint_r0;
				joints[i].localPose0.rotation = joint_q0;
				joints[i].localPose1.position = joint_r1;
				joints[i].localPose1.rotation = joint_q1;

				joints[i].compliance = 0.000000;
				joints[i].rotationXLimited = false;// i == 0 ? false : true;
				//joints[i].minAngleX = -0.01; joints[i].maxAngleX = 0.01;
				joints[i].rotationYLimited = false;
				joints[i].minAngleY = -0.5; joints[i].maxAngleY = 0.5;
				joints[i].rotationZLimited = false;

				root->addPBDJoint(joints[i], rigids[i].second, rigids[i + 1].second);

				/// ********** rigids 1
				Quaternion<float> rigid_q;
				//if (i < 4)
				//	rigid_q = Quaternionf(0, 0, 0.5, 1).normalize();

				//Vector3f rigid_r(0, box_sy * (N - i) - box_sy / 2.0, 0);
				Vector3f rigid_r;
				if (rigids[i].first)
				{
					rigid_r = rigids[i].first->getGlobalR()
						+ rigids[i].first->getGlobalQ().rotate(Vector3f(joint_r0[0], joint_r0[1], joint_r0[2]));
				}
				else
				{
					rigid_r = Vector3f(joint_r0[0], joint_r0[1], joint_r0[2]);
				}
				rigid_r -= rigid_q.rotate(Vector3f(joint_r1[0], joint_r1[1], joint_r1[2]));

				rigids[i + 1].first->setGlobalR(rigid_r);
				rigids[i + 1].first->setGlobalQ(rigid_q);
				//rigids[i + 1].first->setGeometrySize(box_size[0], box_sy, box_sz);
				//rigids[i + 1].first->setI(Inertia<float>(rigid_mass, Vector3f(ixx, iyy, izz)));

				//if (i != 0)
				{
					rigids[i + 1].first->setLinearVelocity(Vector3f(1.0, 0.0, 0.0));
				}

				//if (false)
				//{
				//	/// set velocity
				//	SpatialVector<float> relv(u(e), u(e), u(e), 0, 0, 0);												///< relative velocity, in successor frame.
				//	joint[i]->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[idx_map[id]]));				///< map relative velocity into joint space vector.
				//	motion_state->m_v[id] = (joint[i]->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id]])));	///< set 
				//}
			}
		}
	}


	//// Energy computation module.
	//std::shared_ptr<PendulumExtensionComputeModule> extComputer = std::make_shared<PendulumExtensionComputeModule>();
	//extComputer->outfile = std::string("D:\\Projects\\physiKA\\PostProcess\\PendulumExtension_") +
	//	(useGPU ? std::string("_GPU") : std::string("_CPU_")) + std::to_string(N) +
	//	std::string("_step")+ std::to_string(root->getSolver()->m_numSubstep)+ ".txt";
	//extComputer->maxStep = 600;
	//auto& allrigids = root->getSolver()->getRigidBodys();
	//auto plast = allrigids[allrigids.size() - 1];
	//extComputer->pLastRigid = plast;
	//extComputer->offsetPos = plast->getGlobalR();
	//extComputer->totalLength = box_size[1] * N;
	//root->addCustomModule(extComputer);

	GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
}





DemoContactTest* DemoContactTest::m_instance = 0;
void DemoContactTest::build(bool useGPU)
{
	int N = 50;
	int M = 5;
	double dt = 0.016;

	int ny = 1024, nx = 1024;

	float dl = 0.1;
	float hoffset = 0.1;

	SceneGraph& scene = SceneGraph::getInstance();
	std::shared_ptr<HeightFieldPBDInteractionNode> root = scene.createNewScene<HeightFieldPBDInteractionNode>();
	root->setDt(dt);
	root->setSize(nx, ny, dl, dl);
	root->getSolver()->setUseGPU(useGPU);
	root->getSolver()->m_numSubstep = 10;
	root->getSolver()->m_numContactSolveIter = 10;
	root->getSolver()->setBodyDirty();
	//root->getSolver()->enableVelocitySolve();
	root->getSolver()->disableVelocitySolve();


	HostHeightField1d height;
	height.resize(nx, ny);
	memset(height.GetDataPtr(), 0, sizeof(double) * height.Pitch() * ny);
	for (int i = 0; i < nx; ++i)
	{
		for (int j = 0; j < ny; ++j)
		{
			float curh;
			curh = hoffset;
			height(i, j) = curh;
		}
	}

	DeviceHeightField1d& terrain = root->getHeightField();
	DeviceHeightField1d* terrain_ = &terrain;
	Function1Pt::copy(*terrain_, height);
	root->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);
	//terrain.setOrigin(5, 0, 5);
	{
		auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
		root->addChild(landrigid);

		// Mesh triangles.
		auto triset = std::make_shared< TriangleSet<DataType3f>>();
		landrigid->setTopologyModule(triset);

		// Generate mesh.
		auto& hfland = terrain;
		HeightFieldMesh hfmesh;
		hfmesh.generate(triset, hfland);

		// Mesh renderer.
		auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
		renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
		landrigid->addVisualModule(renderModule);
	}


	Vector3f box_size(0.1, 0.1, 0.1);
	float rigid_mass = 1;
	Vector3f rigid_inertia = RigidUtil::calculateCubeLocalInertia(rigid_mass, box_size);

	for (int mi = 0; mi < M; ++mi)
	{
		for (int mj = 0; mj < M; ++mj)
		{

			/// rigids body
			auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
			int id = root->addRigid(prigid);
			prigid->setDt(dt);
			prigid->setI(Inertia<float>(rigid_mass, rigid_inertia));

			auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
			renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
			prigid->addVisualModule(renderModule);

			prigid->loadShape("../../Media/standard/standard_cube.obj");
			(std::dynamic_pointer_cast<TriangleSet<DataType3f>>)(prigid->getTopologyModule())->scale(box_size / 2.0);
			prigid->setExternalForce(Vector3f(0, -9 * rigid_mass, 0));

			prigid->setLinearDamping(0.5);
			prigid->setAngularDamping(0.5);


			std::default_random_engine e(time(0));
			std::uniform_real_distribution<float> u(-2.0, 2.0);
			std::uniform_real_distribution<float> u1(-1, 1);



			/// ********** rigids
			float dx = 0.3;
			float px = dx * (mi - M / 2);
			float py = 0.06+ hoffset;
			float pz = dx * (mj - M / 2);
			Quaternion<float> rigid_q;

			Vector3f rigid_r(px, py, pz);
			
			

			prigid->setGlobalR(rigid_r);
			prigid->setGlobalQ(rigid_q);

			////if (i != 0)
			//{
			//	rigids[i + 1].first->setLinearVelocity(Vector3f(1.0, 0.0, 0.0));
			//}

			//if (false)
			//{
			//	/// set velocity
			//	SpatialVector<float> relv(u(e), u(e), u(e), 0, 0, 0);												///< relative velocity, in successor frame.
			//	joint[i]->getJointSpace().transposeMul(relv, &(motion_state->generalVelocity[idx_map[id]]));				///< map relative velocity into joint space vector.
			//	motion_state->m_v[id] = (joint[i]->getJointSpace().mul(&(motion_state->generalVelocity[idx_map[id]])));	///< set 
			//}

		}
	}


	auto& camera_ = this->activeCamera();
	Vector3f camPos(0, 10, 5);
	camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));

	GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
}

