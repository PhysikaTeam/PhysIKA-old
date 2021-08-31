#include "demoCar.h"

#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/FreeJoint.h"
#include "Rendering/RigidMeshRender.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Dynamics/RigidBody/Vehicle/SimpleCar.h"
#include "Dynamics/RigidBody/Vehicle/VehicleFrontJoint.h"
#include "Dynamics/RigidBody/Vehicle/VehicleRearJoint.h"

#include "Dynamics/HeightField/HeightFieldMesh.h"

#include "Core/Typedef.h"

#include "Core/Utility/Function1Pt.h"

#include "IO/Image_IO/HeightFieldLoader.h"

#include "TestRigidUtil.h"

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <string>
#include <fstream>

DemoCar::DemoCar()
{
}

void DemoCar::build()
{
    SceneGraph&                                 scene = SceneGraph::getInstance();
    std::shared_ptr<StaticBoundary<DataType3f>> root  = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    //scene.setRootNode(root);

    m_car = std::make_shared<SimpleCar>();
    root->addChild(m_car);

    m_car->carPosition = Vector3f(0, 0.3, 0);
    m_car->carRotation = Quaternion<float>(0, 0, 0, 1);

    m_car->wheelRelPosition[0] = Vector3f(-0.15f, -0.1, 0.2f);
    m_car->wheelRelPosition[1] = Vector3f(+0.15f, -0.1, 0.2f);
    m_car->wheelRelPosition[2] = Vector3f(-0.15f, -0.1, -0.2f);
    m_car->wheelRelPosition[3] = Vector3f(+0.15f, -0.1, -0.2f);
    m_car->wheelRelRotation[0] = Quaternion<float>(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[1] = Quaternion<float>(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[2] = Quaternion<float>(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[3] = Quaternion<float>(0, 0.5, 0, 0.5).normalize();

    m_car->wheelupDirection    = Vector3f(0, 1, 0);
    m_car->wheelRightDirection = Vector3f(1, 0, 0);

    m_car->chassisFile  = "../../Media/standard/standard_cube.obj";
    m_car->wheelFile[0] = "../../Media/Cylinder/cylinder.obj";
    m_car->wheelFile[1] = "../../Media/Cylinder/cylinder.obj";
    m_car->wheelFile[2] = "../../Media/Cylinder/cylinder.obj";
    m_car->wheelFile[3] = "../../Media/Cylinder/cylinder.obj";

    m_car->chassisMeshScale  = Vector3f(0.3, 0.2, 0.5);
    m_car->wheelMeshScale[0] = Vector3f(0.01, 0.01, 0.002);
    m_car->wheelMeshScale[1] = Vector3f(0.01, 0.01, 0.002);
    m_car->wheelMeshScale[2] = Vector3f(0.01, 0.01, 0.002);
    m_car->wheelMeshScale[3] = Vector3f(0.01, 0.01, 0.002);

    m_car->chassisMeshTranslate  = Vector3f(0, 0, 0);
    m_car->wheelMeshTranslate[0] = Vector3f(0, 0, 0.075);
    m_car->wheelMeshTranslate[1] = Vector3f(0, 0, 0.075);
    m_car->wheelMeshTranslate[2] = Vector3f(0, 0, 0.075);
    m_car->wheelMeshTranslate[3] = Vector3f(0, 0, 0.075);

    m_car->chassisMass    = 12;
    m_car->chassisInertia = RigidUtil::calculateCubeLocalInertia(m_car->chassisMass, m_car->chassisMeshScale);

    float    wheelm        = 1;
    Vector3f wheelI        = RigidUtil::calculateCylinderLocalInertia(wheelm, 0.1f, 0.03f, 0);
    m_car->wheelMass[0]    = wheelm;
    m_car->wheelInertia[0] = wheelI;
    m_car->wheelMass[1]    = wheelm;
    m_car->wheelInertia[1] = wheelI;
    m_car->wheelMass[2]    = wheelm;
    m_car->wheelInertia[2] = wheelI;
    m_car->wheelMass[3]    = wheelm;
    m_car->wheelInertia[3] = wheelI;

    m_car->steeringLowerBound = 0;
    m_car->steeringUpperBound = 0;

    m_car->build();

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

DemoCar2* DemoCar2::m_instance = 0;

void DemoCar2::advance(Real dt)
{
    //m_totalTime += dt;
    //float force[6] = { 0 };
    //force[5] = cos(m_totalTime * 0.5);

    //auto pjoint = m_chassis->getParentJoint();
    //pjoint->setMotorForce(force);
}

void DemoCar2::build(bool useGPU)
{
    SceneGraph&                                 scene = SceneGraph::getInstance();
    std::shared_ptr<StaticBoundary<DataType3f>> root  = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    //int ny = 509, nx = 266;
    int ny = 1057, nx = 1057;

    float hScale = 0.2;

    float dl      = 0.22 * hScale;
    float hOffset = -25;

    m_groundRigidInteractor = std::make_shared<HeightFieldPBDInteractionNode>();
    //m_groundRigidInteractor->setRigidBodySystem(m_car->m_rigidSystem);
    m_groundRigidInteractor->setSize(nx, ny, dl, dl);
    m_groundRigidInteractor->getSolver()->m_numSubstep          = 5;
    m_groundRigidInteractor->getSolver()->m_numContactSolveIter = 20;

    m_groundRigidInteractor->getSolver()->setUseGPU(useGPU);

    root->addChild(m_groundRigidInteractor);
    m_groundRigidInteractor->setDt(0.016);

    //m_groundRigidInteractor->setTerrainInfo(terraininfo);

    HostHeightField1d height;
    height.resize(nx, ny);
    memset(height.GetDataPtr(), 0, sizeof(double) * nx * ny);

    //HeightFieldLoader hfloader;
    //hfloader.setRange(0, 0.5);
    //hfloader.load(height, "../../Media/HeightFieldImg/terrain3.png");

    std::string   infilename("../../Media/HeightFieldImg/dem_w1057_h1057.txt");
    std::ifstream infile(infilename);
    if (infile.is_open())
    {
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                float curh;
                infile >> curh;
                curh         = (curh + hOffset) * hScale;
                height(i, j) = curh;
            }
        }
    }
    else
    {
        std::cout << "Open file:  " << infilename.c_str() << "  Failed." << std::endl;
    }

    DeviceHeightField1d& terrain  = m_groundRigidInteractor->getHeightField();
    DeviceHeightField1d* terrain_ = &terrain;
    Function1Pt::copy(*terrain_, height);
    m_groundRigidInteractor->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);
    //m_groundRigidInteractor->setDetectionMethod(HeightFieldTerrainRigidInteractionNode::HFDETECTION::FACEVISE);

    terrain.setOrigin(0, 0, 0);

    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = terrain;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    m_car = std::make_shared<PBDCar>();
    this->addCar(m_car, Vector3f(0.1, /*-0.15*/ 0.90, 0.1), 1, 4 | 8, 2, 4 | 8);

    {
        auto pointset = TypeInfo::cast<PointSet<DataType3f>>(m_car->getChassis()->getTopologyModule());
        if (pointset)
        {
            //Vector3f chaCenter;
            //Vector3f chaSize;

            std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
            //pobb->center = chaCenter;
            //pobb->extent = chaSize;
            pobb->u = Vector3f(1, 0, 0);
            pobb->v = Vector3f(0, 1, 0);
            pobb->w = Vector3f(0, 0, 1);

            //DeviceArray<Vector3f>& vertices = pointset->getPoints();
            this->computeAABB(pointset, pobb->center, pobb->extent);

            auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
            pdetector->addCollidableObject(m_car->m_chassis, pobb);
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        auto pwheel   = m_car->m_wheels[i];
        auto pointset = TypeInfo::cast<PointSet<DataType3f>>(pwheel->getTopologyModule());
        if (pointset)
        {
            //Vector3f chaCenter;
            //Vector3f chaSize;

            std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
            //pobb->center = chaCenter;
            //pobb->extent = chaSize;
            pobb->u = Vector3f(1, 0, 0);
            pobb->v = Vector3f(0, 1, 0);
            pobb->w = Vector3f(0, 0, 1);

            //DeviceArray<Vector3f>& vertices = pointset->getPoints();
            this->computeAABB(pointset, pobb->center, pobb->extent);

            auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
            pdetector->addCollidableObject(pwheel, pobb);
        }
    }

    auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_chassis->getTransformationFrame());
    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
    m_car->m_chassis->addVisualModule(renderModule);
    for (int i = 0; i < 4; ++i)
    {
        auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_wheels[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        m_car->m_wheels[i]->addVisualModule(renderModule);
    }

    m_groundRigidInteractor->addChild(m_car);

    m_car2 = std::make_shared<PBDCar>();
    this->addCar(m_car2, Vector3f(0.1, 0.9, -0.7), 4, 1 | 2, 8, 1 | 2);

    {
        auto pointset = TypeInfo::cast<PointSet<DataType3f>>(m_car2->getChassis()->getTopologyModule());
        if (pointset)
        {
            //Vector3f chaCenter;
            //Vector3f chaSize;

            std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
            //pobb->center = chaCenter;
            //pobb->extent = chaSize;
            pobb->u = Vector3f(1, 0, 0);
            pobb->v = Vector3f(0, 1, 0);
            pobb->w = Vector3f(0, 0, 1);

            //DeviceArray<Vector3f>& vertices = pointset->getPoints();
            this->computeAABB(pointset, pobb->center, pobb->extent);

            auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
            pdetector->addCollidableObject(m_car2->m_chassis, pobb);
        }
    }
    for (int i = 0; i < 4; ++i)
    {
        auto pwheel = m_car2->m_wheels[i];

        auto pointset = TypeInfo::cast<PointSet<DataType3f>>(pwheel->getTopologyModule());
        if (pointset)
        {
            //Vector3f chaCenter;
            //Vector3f chaSize;

            std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
            //pobb->center = chaCenter;
            //pobb->extent = chaSize;
            pobb->u = Vector3f(1, 0, 0);
            pobb->v = Vector3f(0, 1, 0);
            pobb->w = Vector3f(0, 0, 1);

            //DeviceArray<Vector3f>& vertices = pointset->getPoints();
            this->computeAABB(pointset, pobb->center, pobb->extent);

            auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
            pdetector->addCollidableObject(pwheel, pobb);
        }
    }

    renderModule = std::make_shared<RigidMeshRender>(m_car2->m_chassis->getTransformationFrame());
    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
    m_car2->m_chassis->addVisualModule(renderModule);
    for (int i = 0; i < 4; ++i)
    {
        auto renderModule = std::make_shared<RigidMeshRender>(m_car2->m_wheels[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        m_car2->m_wheels[i]->addVisualModule(renderModule);
    }
    m_groundRigidInteractor->addChild(m_car2);

    // Inertaction

    //TerrainRigidInteractionInfo terraininfo;
    //terraininfo.elasticModulus = 1e6;
    //terraininfo.surfaceThickness = 0.05;
    //terraininfo.damping = 5e5;

    //root->addChild(m_car);
    //GLApp window;
    this->setKeyboardFunction(DemoCar2::demoKeyboardFunction);
    this->createWindow(1024, 768);
    this->mainLoop();
}

void DemoCar2::computeAABB(std::shared_ptr<PointSet<DataType3f>> points, Vector3f& center, Vector3f& halfSize)
{
    int nPoints = points->getPointSize();
    if (nPoints <= 0)
        return;

    auto&               pointArr = points->getPoints();
    HostArray<Vector3f> hpoints;
    hpoints.resize(nPoints);
    PhysIKA::Function1Pt::copy(hpoints, pointArr);

    Vector3f pmin = hpoints[0];
    Vector3f pmax = hpoints[0];
    for (int i = 1; i < nPoints; ++i)
    {
        Vector3f curp = hpoints[i];
        pmin[0]       = min(pmin[0], curp[0]);
        pmin[1]       = min(pmin[1], curp[1]);
        pmin[2]       = min(pmin[2], curp[2]);
        pmax[0]       = max(pmax[0], curp[0]);
        pmax[1]       = max(pmax[1], curp[1]);
        pmax[2]       = max(pmax[2], curp[2]);
    }

    center   = (pmin + pmax) * 0.5;
    halfSize = (pmax - pmin) * 0.5;
}

void DemoCar2::addCar(std::shared_ptr<PBDCar> car, Vector3f pos, int chassisGroup, int chassisMask, int wheelGroup, int wheelMask)
{
    car->m_rigidSolver = m_groundRigidInteractor->getSolver();
    //m_car->m_rigidSolver->setUseGPU(useGPU);

    car->carPosition = pos;  // Vector3f(0.1, 0.4, 0.1);
    car->carRotation = Quaternion<float>(0, 0, 0., 1).normalize();

    car->wheelRelPosition[0] = Vector3f(-0.15f /*+ 0.02*/, -0.1, -0.2f /* -0.01*/);
    car->wheelRelPosition[1] = Vector3f(+0.15f /*+0.01*/, -0.1, -0.2f /* +0.02*/);
    car->wheelRelPosition[2] = Vector3f(-0.15f, -0.1, 0.2f);
    car->wheelRelPosition[3] = Vector3f(+0.15f, -0.1, 0.2f);
    car->wheelRelRotation[0] = Quaternion<float>(0, 0, 0, 1);  // (0, 0.5, 0, 0.5).normalize();
    car->wheelRelRotation[1] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    car->wheelRelRotation[2] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    car->wheelRelRotation[3] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();

    car->wheelupDirection    = Vector3f(0, 1, 0);
    car->wheelRightDirection = Vector3f(1, 0, 0);

    car->chassisFile  = "../../Media/standard/standard_cube.obj";
    car->wheelFile[0] = "../../Media/Cylinder/cylinder2.obj";
    car->wheelFile[1] = "../../Media/Cylinder/cylinder2.obj";
    car->wheelFile[2] = "../../Media/Cylinder/cylinder2.obj";
    car->wheelFile[3] = "../../Media/Cylinder/cylinder2.obj";

    car->chassisMeshScale  = Vector3f(0.3, 0.2, 0.5) * 0.5;
    car->wheelMeshScale[0] = Vector3f(0.002, 0.01, 0.01) * 0.5;
    car->wheelMeshScale[1] = Vector3f(0.002, 0.01, 0.01) * 0.5;
    car->wheelMeshScale[2] = Vector3f(0.002, 0.01, 0.01) * 0.5;
    car->wheelMeshScale[3] = Vector3f(0.002, 0.01, 0.01) * 0.5;

    car->wheelRadius[0] = 0.15;
    car->wheelRadius[1] = 0.15;
    car->wheelRadius[2] = 0.15;
    car->wheelRadius[3] = 0.15;

    car->chassisMeshTranslate  = Vector3f(0, 0, 0);
    car->wheelMeshTranslate[0] = Vector3f(0, 0, 0);  // 0.075);
    car->wheelMeshTranslate[1] = Vector3f(0, 0, 0);  // 0.075);
    car->wheelMeshTranslate[2] = Vector3f(0, 0, 0);  // 0.075);
    car->wheelMeshTranslate[3] = Vector3f(0, 0, 0);  // 0.075);

    car->chassisMass    = 5000;  // 00;
    car->chassisInertia = RigidUtil::calculateCubeLocalInertia(car->chassisMass, car->chassisMeshScale);

    float    wheelm      = 50;
    Vector3f wheelI      = RigidUtil::calculateCylinderLocalInertia(wheelm, 0.1f, 0.03f, 0);
    car->wheelMass[0]    = wheelm;
    car->wheelInertia[0] = wheelI;
    car->wheelMass[1]    = wheelm;
    car->wheelInertia[1] = wheelI;
    car->wheelMass[2]    = wheelm;
    car->wheelInertia[2] = wheelI;
    car->wheelMass[3]    = wheelm;
    car->wheelInertia[3] = wheelI;

    car->steeringLowerBound = -0.5;
    car->steeringUpperBound = 0.5;

    car->forwardForceAcc = 10000;
    //car->breakForceAcc ;
    car->steeringSpeed = 1.0;
    car->maxVel        = 2.5;

    car->chassisCollisionGroup = chassisGroup;
    car->chassisCollisionMask  = chassisMask;
    car->wheelCollisionGroup   = wheelGroup;
    car->wheelCollisionMask    = wheelMask;

    car->linearDamping  = 0.2;
    car->angularDamping = 0.2;

    car->suspensionLength   = 0.05;
    car->suspensionStrength = 1000000;

    car->build();
    car->setDt(0.016);
}

void DemoCar2::demoKeyboardFunction(unsigned char key, int x, int y)
{
    if (key != 's' && key != 'a' && key != 'd' && key != 'w')
    {
        GLApp::keyboardFunction(key, x, y);
    }
    else
    {
        if (!m_instance)
            return;
        switch (key)
        {
            case 'a':
                m_instance->m_car->goLeft(0.016);
                break;
            case 'd':
                m_instance->m_car->goRight(0.016);
                break;
            case 'w':
                m_instance->m_car->forward(0.016);
                break;
            case 's':
                m_instance->m_car->backward(0.016);
                break;
        }
    }
}

DemoTankCar* DemoTankCar::m_instance = 0;

void DemoTankCar::build(bool useGPU)
{
    SceneGraph&                                 scene = SceneGraph::getInstance();
    std::shared_ptr<StaticBoundary<DataType3f>> root  = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    //int ny = 509, nx = 266;
    int ny = 1057, nx = 1057;

    float hScale = 0.4;

    float dl      = 0.22 * hScale;
    float hOffset = -26;

    m_groundRigidInteractor = std::make_shared<HeightFieldPBDInteractionNode>();
    //m_groundRigidInteractor->setRigidBodySystem(m_car->m_rigidSystem);
    m_groundRigidInteractor->setSize(nx, ny, dl, dl);
    m_groundRigidInteractor->getSolver()->m_numSubstep          = 4;
    m_groundRigidInteractor->getSolver()->m_numContactSolveIter = 30;

    m_groundRigidInteractor->getSolver()->setUseGPU(useGPU);

    root->addChild(m_groundRigidInteractor);
    m_groundRigidInteractor->setDt(0.016);

    //m_groundRigidInteractor->setTerrainInfo(terraininfo);

    HostHeightField1d height;
    height.resize(nx, ny);
    memset(height.GetDataPtr(), 0, sizeof(double) * nx * ny);

    //HeightFieldLoader hfloader;
    //hfloader.setRange(0, 0.5);
    //hfloader.load(height, "../../Media/HeightFieldImg/terrain3.png");

    std::string   infilename("../../Media/HeightFieldImg/dem_w1057_h1057.txt");
    std::ifstream infile(infilename);
    if (infile.is_open())
    {
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                float curh;
                infile >> curh;
                curh         = (curh + hOffset) * hScale;
                height(i, j) = curh;
            }
        }
    }
    else
    {
        std::cout << "Open file:  " << infilename.c_str() << "  Failed." << std::endl;
    }

    DeviceHeightField1d& terrain  = m_groundRigidInteractor->getHeightField();
    DeviceHeightField1d* terrain_ = &terrain;
    Function1Pt::copy(*terrain_, height);
    m_groundRigidInteractor->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);
    //m_groundRigidInteractor->setDetectionMethod(HeightFieldTerrainRigidInteractionNode::HFDETECTION::FACEVISE);

    terrain.setOrigin(0, 0, 0);

    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = terrain;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    m_car = std::make_shared<MultiWheelCar<4>>();
    this->addCar(m_car, Vector3f(0.1, /*-0.15*/ 0.40, 0.1), 1, 4 | 8, 2, 4 | 8);

    {
        auto pointset = TypeInfo::cast<PointSet<DataType3f>>(m_car->getChassis()->getTopologyModule());
        if (pointset)
        {
            //Vector3f chaCenter;
            //Vector3f chaSize;

            std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
            //pobb->center = chaCenter;
            //pobb->extent = chaSize;
            pobb->u = Vector3f(1, 0, 0);
            pobb->v = Vector3f(0, 1, 0);
            pobb->w = Vector3f(0, 0, 1);

            //DeviceArray<Vector3f>& vertices = pointset->getPoints();
            this->computeAABB(pointset, pobb->center, pobb->extent);

            auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
            pdetector->addCollidableObject(m_car->m_chassis, pobb);
        }
    }
    for (int lr = 0; lr < 2; ++lr)
    {
        for (int i = 0; i < 4; ++i)
        {
            auto pwheel   = m_car->m_wheels[lr][i];
            auto pointset = TypeInfo::cast<PointSet<DataType3f>>(pwheel->getTopologyModule());
            if (pointset)
            {
                //Vector3f chaCenter;
                //Vector3f chaSize;

                std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
                //pobb->center = chaCenter;
                //pobb->extent = chaSize;
                pobb->u = Vector3f(1, 0, 0);
                pobb->v = Vector3f(0, 1, 0);
                pobb->w = Vector3f(0, 0, 1);

                //DeviceArray<Vector3f>& vertices = pointset->getPoints();
                this->computeAABB(pointset, pobb->center, pobb->extent);

                auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
                pdetector->addCollidableObject(pwheel, pobb);
            }
        }
    }

    auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_chassis->getTransformationFrame());
    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
    m_car->m_chassis->addVisualModule(renderModule);
    for (int lr = 0; lr < 2; ++lr)
    {
        for (int i = 0; i < 4; ++i)
        {
            auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_wheels[lr][i]->getTransformationFrame());
            renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
            m_car->m_wheels[lr][i]->addVisualModule(renderModule);
        }
    }
    m_groundRigidInteractor->addChild(m_car);

    //m_car2 = std::make_shared<PBDCar>();
    //this->addCar(m_car2, Vector3f(0.1, 0.4, -0.7),
    //	4, 1 | 2, 8, 1 | 2
    //);

    //{
    //	auto pointset = TypeInfo::cast<PointSet<DataType3f>>(m_car2->getChassis()->getTopologyModule());
    //	if (pointset)
    //	{
    //		//Vector3f chaCenter;
    //		//Vector3f chaSize;

    //		std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
    //		//pobb->center = chaCenter;
    //		//pobb->extent = chaSize;
    //		pobb->u = Vector3f(1, 0, 0);
    //		pobb->v = Vector3f(0, 1, 0);
    //		pobb->w = Vector3f(0, 0, 1);

    //		//DeviceArray<Vector3f>& vertices = pointset->getPoints();
    //		computeMeshAABB(pointset, pobb->center, pobb->extent);

    //		auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
    //		pdetector->addCollidableObject(m_car2->m_chassis, pobb);

    //	}
    //}
    //for (int lr = 0; lr < 2; ++lr)
    //{
    //	for (int i = 0; i < 4; ++i)
    //	{
    //		auto pwheel = m_car2->m_wheels[lr][i];

    //		auto pointset = TypeInfo::cast<PointSet<DataType3f>>(pwheel->getTopologyModule());
    //		if (pointset)
    //		{
    //			//Vector3f chaCenter;
    //			//Vector3f chaSize;

    //			std::shared_ptr<TOrientedBox3D<float>> pobb = std::make_shared<TOrientedBox3D<float>>();
    //			//pobb->center = chaCenter;
    //			//pobb->extent = chaSize;
    //			pobb->u = Vector3f(1, 0, 0);
    //			pobb->v = Vector3f(0, 1, 0);
    //			pobb->w = Vector3f(0, 0, 1);

    //			//DeviceArray<Vector3f>& vertices = pointset->getPoints();
    //			computeMeshAABB(pointset, pobb->center, pobb->extent);

    //			auto pdetector = m_groundRigidInteractor->getRigidContactDetector();
    //			pdetector->addCollidableObject(pwheel, pobb);

    //		}
    //	}
    //}

    //renderModule = std::make_shared<RigidMeshRender>(m_car2->m_chassis->getTransformationFrame());
    //renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
    //m_car2->m_chassis->addVisualModule(renderModule);
    //for (int lr = 0; lr < 2; ++lr)
    //{
    //	for (int i = 0; i < 4; ++i)
    //	{
    //		auto renderModule = std::make_shared<RigidMeshRender>(m_car2->m_wheels[lr][i]->getTransformationFrame());
    //		renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
    //		m_car2->m_wheels[lr][i]->addVisualModule(renderModule);
    //	}
    //}
    //m_groundRigidInteractor->addChild(m_car2);

    // Inertaction

    //TerrainRigidInteractionInfo terraininfo;
    //terraininfo.elasticModulus = 1e6;
    //terraininfo.surfaceThickness = 0.05;
    //terraininfo.damping = 5e5;

    //root->addChild(m_car);
    //GLApp window;
    this->setKeyboardFunction(DemoTankCar::demoKeyboardFunction);
    this->createWindow(1024, 768);
    this->mainLoop();
}

void DemoTankCar::addCar(std::shared_ptr<MultiWheelCar<4>> car, Vector3f pos, int chassisGroup, int chassisMask, int wheelGroup, int wheelMask)
{
    car->m_rigidSolver = m_groundRigidInteractor->getSolver();
    //m_car->m_rigidSolver->setUseGPU(useGPU);

    car->carPosition = pos;  // Vector3f(0.1, 0.4, 0.1);
    car->carRotation = Quaternion<float>(0, 0, 0., 1).normalize();

    car->chassisFile      = "../../Media/standard/standard_cube.obj";
    car->chassisMeshScale = Vector3f(0.3, 0.2, 0.5) * 0.5;

    car->chassisMeshTranslate = Vector3f(0, 0, 0);

    car->chassisMass    = 5000;  // 00;
    car->chassisInertia = RigidUtil::calculateCubeLocalInertia(car->chassisMass, car->chassisMeshScale);

    for (int lr = 0; lr < 2; ++lr)
    {
        float sx                = 0.3f;
        car->wheelRelPos[lr][0] = Vector3f(-0.5 * sx + lr * sx, -0.1 + 0.02, -0.3);
        car->wheelRelPos[lr][1] = Vector3f(-0.5 * sx + lr * sx, -0.1, -0.1);
        car->wheelRelPos[lr][2] = Vector3f(-0.5 * sx + lr * sx, -0.1, 0.1);
        car->wheelRelPos[lr][3] = Vector3f(-0.5 * sx + lr * sx, -0.1 + 0.02, 0.3);

        car->wheelRelRot[lr][0] = Quaternion<float>(0, 0, 0, 1);
        car->wheelRelRot[lr][1] = Quaternion<float>(0, 0, 0, 1);
        car->wheelRelRot[lr][2] = Quaternion<float>(0, 0, 0, 1);
        car->wheelRelRot[lr][3] = Quaternion<float>(0, 0, 0, 1);

        car->wheelFile[lr][0]      = "../../Media/Cylinder/cylinder2.obj";
        car->wheelFile[lr][1]      = "../../Media/Cylinder/cylinder2.obj";
        car->wheelFile[lr][2]      = "../../Media/Cylinder/cylinder2.obj";
        car->wheelFile[lr][3]      = "../../Media/Cylinder/cylinder2.obj";
        car->wheelMeshScale[lr][0] = Vector3f(0.002, 0.01, 0.01) * 0.5;
        car->wheelMeshScale[lr][1] = Vector3f(0.002, 0.01, 0.01) * 0.5;
        car->wheelMeshScale[lr][2] = Vector3f(0.002, 0.01, 0.01) * 0.5;
        car->wheelMeshScale[lr][3] = Vector3f(0.002, 0.01, 0.01) * 0.5;

        car->wheelMeshTranslate[lr][0] = Vector3f(0, 0, 0);  // 0.075);
        car->wheelMeshTranslate[lr][1] = Vector3f(0, 0, 0);  // 0.075);
        car->wheelMeshTranslate[lr][2] = Vector3f(0, 0, 0);  // 0.075);
        car->wheelMeshTranslate[lr][3] = Vector3f(0, 0, 0);  // 0.075);

        float    wheelm = 50;
        Vector3f wheelI = RigidUtil::calculateCylinderLocalInertia(wheelm, 0.1f, 0.03f, 0);
        printf("%f  %f  %f\n", wheelI[0], wheelI[1], wheelI[2]);
        car->wheelMass[lr][0]    = wheelm;
        car->wheelInertia[lr][0] = wheelI;
        car->wheelMass[lr][1]    = wheelm;
        car->wheelInertia[lr][1] = wheelI;
        car->wheelMass[lr][2]    = wheelm;
        car->wheelInertia[lr][2] = wheelI;
        car->wheelMass[lr][3]    = wheelm;
        car->wheelInertia[lr][3] = wheelI;
    }

    car->upDirection    = Vector3f(0, 1, 0);
    car->rightDirection = Vector3f(1, 0, 0);

    car->forwardForceAcc = 1000;
    car->maxVel          = 2.5;

    car->chassisCollisionGroup = chassisGroup;
    car->chassisCollisionMask  = chassisMask;
    car->wheelCollisionGroup   = wheelGroup;
    car->wheelCollisionMask    = wheelMask;

    car->linearDamping  = 0.8;
    car->angularDamping = 0.8;

    car->suspensionLength   = 0.05;
    car->suspensionStrength = 1000000;

    car->build();
    car->setDt(0.016);
}

void DemoTankCar::computeAABB(std::shared_ptr<PointSet<DataType3f>> points, Vector3f& center, Vector3f& halfSize)
{
    int nPoints = points->getPointSize();
    if (nPoints <= 0)
        return;

    auto&               pointArr = points->getPoints();
    HostArray<Vector3f> hpoints;
    hpoints.resize(nPoints);
    PhysIKA::Function1Pt::copy(hpoints, pointArr);

    Vector3f pmin = hpoints[0];
    Vector3f pmax = hpoints[0];
    for (int i = 1; i < nPoints; ++i)
    {
        Vector3f curp = hpoints[i];
        pmin[0]       = min(pmin[0], curp[0]);
        pmin[1]       = min(pmin[1], curp[1]);
        pmin[2]       = min(pmin[2], curp[2]);
        pmax[0]       = max(pmax[0], curp[0]);
        pmax[1]       = max(pmax[1], curp[1]);
        pmax[2]       = max(pmax[2], curp[2]);
    }

    center   = (pmin + pmax) * 0.5;
    halfSize = (pmax - pmin) * 0.5;
}

void DemoTankCar::demoKeyboardFunction(unsigned char key, int x, int y)
{
    if (key != 's' && key != 'a' && key != 'd' && key != 'w')
    {
        GLApp::keyboardFunction(key, x, y);
    }
    else
    {
        if (!m_instance)
            return;
        switch (key)
        {
            case 'a':
                m_instance->m_car->goLeft(0.016);
                break;
            case 'd':
                m_instance->m_car->goRight(0.016);
                break;
            case 'w':
                m_instance->m_car->forward(0.016);
                break;
            case 's':
                m_instance->m_car->backward(0.016);
                break;
        }
    }
}

DemoPBDCar* DemoPBDCar::m_instance = 0;

void DemoPBDCar::advance(Real dt)
{
    //m_totalTime += dt;
    //float force[6] = { 0 };
    //force[5] = cos(m_totalTime * 0.5);

    //auto pjoint = m_chassis->getParentJoint();
    //pjoint->setMotorForce(force);
}

void DemoPBDCar::build(bool useGPU)
{
    SceneGraph&                                 scene = SceneGraph::getInstance();
    std::shared_ptr<StaticBoundary<DataType3f>> root  = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    int   nx = 128, ny = 128;
    float dl = 0.01;

    m_groundRigidInteractor = std::make_shared<HeightFieldPBDInteractionNode>();
    //m_groundRigidInteractor->setRigidBodySystem(m_car->m_rigidSystem);
    m_groundRigidInteractor->setSize(nx, ny, dl, dl);
    //m_groundRigidInteractor->setTerrainInfo(terraininfo);

    Array2D<double, DeviceType::CPU> height;
    height.resize(nx, ny);
    memset(height.GetDataPtr(), 0, sizeof(float) * nx * ny);

    DeviceHeightField1d& terrain  = m_groundRigidInteractor->getHeightField();
    DeviceHeightField1d* terrain_ = &terrain;
    Function1Pt::copy(*terrain_, height);
    m_groundRigidInteractor->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);
    //m_groundRigidInteractor->setDetectionMethod(HeightFieldTerrainRigidInteractionNode::HFDETECTION::FACEVISE);

    m_car = std::make_shared<PBDCar>();
    //root->addChild(m_car);
    m_car->m_rigidSolver = m_groundRigidInteractor->getSolver();
    m_car->m_rigidSolver->setUseGPU(useGPU);

    m_car->carPosition = Vector3f(0, 0.4, 0);
    m_car->carRotation = Quaternion<float>(0, 0, 0., 1).normalize();

    m_car->wheelRelPosition[0] = Vector3f(-0.15f /*+ 0.02*/, -0.1, -0.2f /* -0.01*/);
    m_car->wheelRelPosition[1] = Vector3f(+0.15f /*+0.01*/, -0.1, -0.2f /* +0.02*/);
    m_car->wheelRelPosition[2] = Vector3f(-0.15f, -0.1, 0.2f);
    m_car->wheelRelPosition[3] = Vector3f(+0.15f, -0.1, 0.2f);
    m_car->wheelRelRotation[0] = Quaternion<float>(0, 0, 0, 1);  // (0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[1] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[2] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[3] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();

    m_car->wheelupDirection    = Vector3f(0, 1, 0);
    m_car->wheelRightDirection = Vector3f(1, 0, 0);

    m_car->chassisFile  = "../../Media/standard/standard_cube.obj";
    m_car->wheelFile[0] = "../../Media/Cylinder/cylinder2.obj";
    m_car->wheelFile[1] = "../../Media/Cylinder/cylinder2.obj";
    m_car->wheelFile[2] = "../../Media/Cylinder/cylinder2.obj";
    m_car->wheelFile[3] = "../../Media/Cylinder/cylinder2.obj";

    m_car->chassisMeshScale  = Vector3f(0.3, 0.2, 0.5) * 0.5;
    m_car->wheelMeshScale[0] = Vector3f(0.002, 0.01, 0.01) * 0.5;
    m_car->wheelMeshScale[1] = Vector3f(0.002, 0.01, 0.01) * 0.5;
    m_car->wheelMeshScale[2] = Vector3f(0.002, 0.01, 0.01) * 0.5;
    m_car->wheelMeshScale[3] = Vector3f(0.002, 0.01, 0.01) * 0.5;

    m_car->wheelRadius[0] = 0.15;
    m_car->wheelRadius[1] = 0.15;
    m_car->wheelRadius[2] = 0.15;
    m_car->wheelRadius[3] = 0.15;

    m_car->chassisMeshTranslate  = Vector3f(0, 0, 0);
    m_car->wheelMeshTranslate[0] = Vector3f(0, 0, 0);  // 0.075);
    m_car->wheelMeshTranslate[1] = Vector3f(0, 0, 0);  // 0.075);
    m_car->wheelMeshTranslate[2] = Vector3f(0, 0, 0);  // 0.075);
    m_car->wheelMeshTranslate[3] = Vector3f(0, 0, 0);  // 0.075);

    m_car->chassisMass    = 1500;  // 00;
    m_car->chassisInertia = RigidUtil::calculateCubeLocalInertia(m_car->chassisMass, m_car->chassisMeshScale);

    float    wheelm        = 50;
    Vector3f wheelI        = RigidUtil::calculateCylinderLocalInertia(wheelm, 0.1f, 0.03f, 0);
    m_car->wheelMass[0]    = wheelm;
    m_car->wheelInertia[0] = wheelI;
    m_car->wheelMass[1]    = wheelm;
    m_car->wheelInertia[1] = wheelI;
    m_car->wheelMass[2]    = wheelm;
    m_car->wheelInertia[2] = wheelI;
    m_car->wheelMass[3]    = wheelm;
    m_car->wheelInertia[3] = wheelI;

    m_car->steeringLowerBound = -0.5;
    m_car->steeringUpperBound = 0.5;

    m_car->forwardForceAcc = 5000;
    //m_car->breakForceAcc ;
    m_car->steeringSpeed = 1.0;
    m_car->maxVel        = 2.5;

    m_car->build();

    auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_chassis->getTransformationFrame());
    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
    m_car->m_chassis->addVisualModule(renderModule);
    for (int i = 0; i < 4; ++i)
    {
        auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_wheels[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        m_car->m_wheels[i]->addVisualModule(renderModule);
    }

    // Inertaction

    //TerrainRigidInteractionInfo terraininfo;
    //terraininfo.elasticModulus = 1e6;
    //terraininfo.surfaceThickness = 0.05;
    //terraininfo.damping = 5e5;

    m_groundRigidInteractor->addChild(m_car);
    root->addChild(m_groundRigidInteractor);

    m_car->setDt(0.016);
    m_groundRigidInteractor->setDt(0.016);

    //root->addChild(m_car);
    //GLApp window;
    this->setKeyboardFunction(DemoPBDCar::demoKeyboardFunction);
    this->createWindow(1024, 768);
    this->mainLoop();
}

void DemoPBDCar::demoKeyboardFunction(unsigned char key, int x, int y)
{
    if (key != 's' && key != 'a' && key != 'd' && key != 'w')
    {
        GLApp::keyboardFunction(key, x, y);
    }
    else
    {
        if (!m_instance)
            return;
        switch (key)
        {
            case 'a':
                m_instance->m_car->goLeft(0.016);
                break;
            case 'd':
                m_instance->m_car->goRight(0.016);
                break;
            case 'w':
                m_instance->m_car->forward(0.016);
                break;
            case 's':
                m_instance->m_car->backward(0.016);
                break;
        }
    }
}

DemoSlope* DemoSlope::m_instance = 0;
void       DemoSlope::build(bool useGPU)
{

    SceneGraph&                                 scene = SceneGraph::getInstance();
    std::shared_ptr<StaticBoundary<DataType3f>> root  = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));

    //int ny = 509, nx = 266;
    int   ny = 64, nx = 64;
    float mu = 0.5;

    float hScale = 0.05;

    float dl      = /*0.22 **/ hScale;
    float hOffset = -25;

    m_groundRigidInteractor = std::make_shared<HeightFieldPBDInteractionNode>();
    root->addChild(m_groundRigidInteractor);

    //m_groundRigidInteractor->setRigidBodySystem(m_car->m_rigidSystem);
    m_groundRigidInteractor->setSize(nx, ny, dl, dl);
    m_groundRigidInteractor->getSolver()->m_numSubstep = 10;
    //m_groundRigidInteractor->setTerrainInfo(terraininfo);

    // land
    HostHeightField1d landHeight;
    landHeight.resize(nx, ny);

    float lhLand = nx * dl / 2.0 * mu + 0.25;
    float dhLand = lhLand / nx * 2;
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            landHeight[i * ny + j] = lhLand - dhLand * j;
            if (landHeight[i * ny + j] < 0)
                landHeight[i * ny + j] = 0.0f;
        }
    }

    DeviceHeightField1d& terrain  = m_groundRigidInteractor->getHeightField();
    DeviceHeightField1d* terrain_ = &terrain;
    Function1Pt::copy(*terrain_, landHeight);
    m_groundRigidInteractor->setDetectionMethod(HeightFieldPBDInteractionNode::HFDETECTION::POINTVISE);

    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = terrain;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    /// ------  Rigid ------------

    double   scale1d = 0.15;
    Vector3f scale(scale1d, scale1d, scale1d);
    double   rhorigid = 2000;
    float    radius   = 1.0;
    radius *= scale1d;
    float    rigid_mass = 1000;
    Vector3f rigidI     = RigidUtil::calculateCubeLocalInertia(
        rigid_mass, scale * 2);

    double      rotRad = atan(dhLand / dl);
    Quaternionf cubeRot(Vector3f(0, 0, -1), rotRad);
    int         N = 1;
    for (int i = 0; i < N; ++i)
    {

        /// rigids body
        auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
        int  id     = m_groundRigidInteractor->addRigid(prigid);

        prigid->setMu(mu);

        auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        prigid->addVisualModule(renderModule);
        //m_rigids.push_back(prigid);
        //m_rigidRenders.push_back(renderModule);

        prigid->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
        //triset->translate(Vector3f(0, 0, -0.5));
        triset->scale(scale);

        //prigid->setGeometrySize(scale[0], scale[1], scale[2]);
        //prigid->setAngularVelocity(Vector3f(0., 0.0, -1.0));

        prigid->setLinearVelocity(Vector3f(-0.0, 0.0, 0));
        prigid->setGlobalR(Vector3f(-0.5 * i - 0.3, /* 0.6*/ 0.32 + 0.1 + 0.5 * i, 0));
        prigid->setGlobalQ(/*Quaternionf(0, 0, 0, 1).normalize()*/ cubeRot);
        prigid->setExternalForce(Vector3f(0, -5 * rigid_mass, 0));
        prigid->setI(Inertia<float>(rigid_mass, rigidI));

        //DistanceField3D<DataType3f> sdf;
        //sdf.loadSDF("../../Media/standard/standard_cube.sdf");
        ////sdf.translate(Vector3f(0, 0, -0.5) );
        //sdf.scale(scale1d);
        //interactionSolver->addSDF(sdf, id);
    }

    m_groundRigidInteractor->setDt(0.016);
    m_groundRigidInteractor->getSolver()->setUseGPU(useGPU);

    //root->addChild(m_car);
    //GLApp window;
    //this->setKeyboardFunction(DemoSlope::demoKeyboardFunction);
    this->createWindow(1024, 768);
    this->mainLoop();
}