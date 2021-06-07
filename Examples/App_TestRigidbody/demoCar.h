#pragma once


#include "GUI/GlutGUI/GLApp.h"
//#include "Framework/Framework/Node.h"

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"

#include "Dynamics/RigidBody/Vehicle/SimpleCar.h"
#include "Dynamics/RigidBody/Vehicle/HeightFieldTerrainRigidInteractionNode.h"
#include "Dynamics/RigidBody/Vehicle/PBDCar.h"
#include "Dynamics/RigidBody/PBDRigid/HeightFieldPBDInteractionNode.h"
#include "Dynamics/RigidBody/Vehicle/MultiWheelCar.h"

using namespace PhysIKA;

class DemoCar// :public Node
{
public:
	DemoCar();

	void build();

	virtual void advance(Real dt) {}
private:

	std::shared_ptr<SimpleCar> m_car;


	std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

	std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

	std::shared_ptr<RigidBody2<DataType3f>> m_wheels[4];

	float m_totalTime = 0.0;
};



class DemoCar2:public GLApp
{
private:
	DemoCar2() {}
	static DemoCar2* m_instance;
public:
	static DemoCar2* getInstance()
	{
		if (m_instance == 0)
			m_instance = new DemoCar2;
		return m_instance;
	}

	void build(bool useGPU = true);

	void run() {}

	virtual void advance(Real dt);

	static void demoKeyboardFunction(unsigned char key, int x, int y);

	void addCar(std::shared_ptr<PBDCar> car, Vector3f pos,
		int chassisGroup=1, int chassisMask=1, int wheelGroup=2, int wheelMask = 4);

	void computeAABB(std::shared_ptr<PointSet<DataType3f>> points, Vector3f& center, Vector3f& halfSize);

private:

	std::shared_ptr<PBDCar> m_car;
	std::shared_ptr<PBDCar> m_car2;

	std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;


	std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

	//std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

	//std::shared_ptr<RigidBody2<DataType3f>> m_wheels[4];

	float m_totalTime = 0.0;
};



class DemoTankCar :public GLApp
{
private:
	DemoTankCar() {}
	static DemoTankCar* m_instance;
public:
	static DemoTankCar* getInstance()
	{
		if (m_instance == 0)
			m_instance = new DemoTankCar();
		return m_instance;
	}

	void build(bool useGPU = true);

	void run() {}

	virtual void advance(Real dt) {}

	static void demoKeyboardFunction(unsigned char key, int x, int y);

	void addCar(std::shared_ptr<MultiWheelCar<4>> car, Vector3f pos,
		int chassisGroup = 1, int chassisMask = 1, int wheelGroup = 2, int wheelMask = 4);

	void computeAABB(std::shared_ptr<PointSet<DataType3f>> points, Vector3f& center, Vector3f& halfSize);

private:

	std::shared_ptr<MultiWheelCar<4>> m_car;
	std::shared_ptr<MultiWheelCar<4>> m_car2;

	std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;


	std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

	//std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

	//std::shared_ptr<RigidBody2<DataType3f>> m_wheels[4];

	float m_totalTime = 0.0;
};


class DemoPBDCar :public GLApp
{
private:
	DemoPBDCar() {}
	static DemoPBDCar* m_instance;
public:
	static DemoPBDCar* getInstance()
	{
		if (m_instance == 0)
			m_instance = new DemoPBDCar;
		return m_instance;
	}

	void build(bool useGPU=true);

	void run() {}

	virtual void advance(Real dt);

	static void demoKeyboardFunction(unsigned char key, int x, int y);

private:

	std::shared_ptr<PBDCar> m_car;
	std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;


	std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

	//std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

	//std::shared_ptr<RigidBody2<DataType3f>> m_wheels[4];

	float m_totalTime = 0.0;
};

//class demoCarApp :public GLApp
//{
//public:
//	demoCarApp();
//
//	/**
//	* @brief Initialization of demo, build up scene graph
//	*/
//	void initialize();
//
//};





class DemoSlope :public GLApp
{
private:
	DemoSlope() {}
	static DemoSlope* m_instance;
public:
	static DemoSlope* getInstance()
	{
		if (m_instance == 0)
			m_instance = new DemoSlope;
		return m_instance;
	}

	void build(bool useGPU = true);

	void run() {}

	//static void demoKeyboardFunction(unsigned char key, int x, int y);

private:

	//std::shared_ptr<PBDCar> m_car;
	std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;


	std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

	//std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

	//std::shared_ptr<RigidBody2<DataType3f>> m_wheels[4];

	float m_totalTime = 0.0;
};