#pragma once

#include "GUI/GlutGUI/GLApp.h"

#include "Dynamics/RigidBody/PBDRigid/HeightFieldPBDInteractionNode.h"

#include <memory>

using namespace PhysIKA;

//template<int N = 10>
class DemoPBDPositionConstraint
{
public:
    DemoPBDPositionConstraint(int n = 10, bool usegpu = true)
        : N(n), useGPU(usegpu) {}
    void run();

public:
    int  N = 10;
    bool useGPU;
};

class DemoPBDRotationConstraint
{
public:
    DemoPBDRotationConstraint(int n = 10, bool usegpu = true)
        : N(n), useGPU(usegpu) {}
    void run();

public:
    int  N = 10;
    bool useGPU;
};

//class DemoPBDCommonRigid
//{
//public:
//       DemoPBDCommonRigid(int n = 10, bool usegpu = true) :N(n), useGPU(usegpu) {}
//       void run();
//
//public:
//       int N = 10;
//       bool useGPU;
//};

class DemoPBDSingleHFCollide : public GLApp
{

private:
    DemoPBDSingleHFCollide()
    {
        createWindow(1024, 768);
    }
    static DemoPBDSingleHFCollide* m_instance;

public:
    static DemoPBDSingleHFCollide* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoPBDSingleHFCollide;
        return m_instance;
    }

    void createScene();

    void run()
    {
        mainLoop();
    }

public:
    //std::vector<RigidBody2_ptr> m_rigids;
    //std::vector<std::shared_ptr<RigidMeshRender>> m_rigidRenders;

    //bool m_rigidVisible = true;
};

class DemoCollisionTest : public GLApp
{
private:
    DemoCollisionTest() {}
    static DemoCollisionTest* m_instance;

public:
    static DemoCollisionTest* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoCollisionTest;
        return m_instance;
    }

    void build(bool useGPU = false);

    void run() {}

    //virtual void advance(Real dt);

    //static void demoKeyboardFunction(unsigned char key, int x, int y);

private:
    std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;

    //std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

    float m_totalTime = 0.0;
};

class DemoPendulumTest : public GLApp
{
private:
    DemoPendulumTest() {}
    static DemoPendulumTest* m_instance;

public:
    static DemoPendulumTest* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoPendulumTest;
        return m_instance;
    }

    void build(bool useGPU = false);

    void run() {}

private:
    //std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;

    //std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

    float m_totalTime = 0.0;
};

class DemoContactTest : public GLApp
{
private:
    DemoContactTest() {}
    static DemoContactTest* m_instance;

public:
    static DemoContactTest* getInstance()
    {
        if (m_instance == 0)
            m_instance = new DemoContactTest;
        return m_instance;
    }

    void build(bool useGPU = false);

    void run() {}

private:
    //std::shared_ptr<HeightFieldPBDInteractionNode> m_groundRigidInteractor;

    //std::shared_ptr<RigidBodyRoot<DataType3f>> m_rigidSystem;

    float m_totalTime = 0.0;
};