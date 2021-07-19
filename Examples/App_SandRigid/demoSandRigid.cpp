#include "demoSandRigid.h"

#include "sandRigidCommon.h"
#include "Dynamics/Sand/SandSimulator.h"
#include "Framework/Framework/SceneGraph.h"
#include "Rendering/PointRenderModule.h"
#include "Dynamics/Sand/PBDSandSolver.h"
#include "Dynamics/Sand/PBDSandRigidInteraction.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolverNode.h"
#include "Rendering/RigidMeshRender.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Dynamics/Sand/ParticleSandRigidInteraction.h"
#include "Dynamics/Sand/HeightFieldSandRigidInteraction.h"
#include "Dynamics/HeightField/HeightFieldMesh.h"
#include "IO/Surface_Mesh_IO/ObjFileLoader.h"

#include "Dynamics/Sand/SandVisualPointSampleModule.h"

#include "IO/Image_IO/HeightFieldLoader.h"
#include "Dynamics/Sand/SSESandSolver.h"
#include "Dynamics/Sand/SandSimulator.h"

#include "sandRigidCommon.h"
#include <random>

DemoHeightFieldSand* DemoHeightFieldSand::m_instance = 0;
void                 DemoHeightFieldSand::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64;
    sandinfo.ny               = 64;
    sandinfo.griddl           = 0.05;
    sandinfo.mu               = tan(30.9 / 180 * 3.14159);  // 0.7;
    sandinfo.drag             = 0.95;
    sandinfo.slide            = .2;  // 10 * sandinfo.griddl;
    sandinfo.sandRho          = 16000.0;
    double sandParticleHeight = 0.1;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(10, 10, 10));
    scene.setLowerBound(Vector3f(-10, -5, -10));

    //// Root node. Also the simulator.
    //std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    //root->setActive(true);
    //root->setDt(0.02);
    //auto interactionSolver = root->getInteractionSolver();

    // Sand simulator.
    std::shared_ptr<SandSimulator> sandSim     = scene.createNewScene<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    sandSim->needForward(true);
    sandSim->setSandSolver(psandSolver);
    m_sandsolver = psandSolver;
    //root->setSandSolver(psandSolver);
    //root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();  //sandSim->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    //psandSolver->updateSandStaticHeight(0.02);
    //psandSolver->stepSimulation(0.01);
    //root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 0, sandinfo.nx, 0, 20 };
    fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.5f);
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.2);
    pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = /*&(m_sandsolver->m_sandStaticHeight);*/ &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 4, 2, sandinfo.griddl);
    m_sampler = psampler;

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 1.5, 5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}

DemoHeightFieldSandRigid_Sphere* DemoHeightFieldSandRigid_Sphere::m_instance = 0;
void                             DemoHeightFieldSandRigid_Sphere::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64 /**1.5*/;
    sandinfo.ny               = 64;
    sandinfo.griddl           = 0.04;
    sandinfo.mu               = 0.7;
    sandinfo.drag             = 0.95;
    sandinfo.slide            = 10 * sandinfo.griddl;
    sandinfo.sandRho          = 1000.0;
    double sandParticleHeight = 0.1;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(10, 10, 10));
    scene.setLowerBound(Vector3f(-10, -5, -10));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    root->m_subStep        = 1;
    auto interactionSolver = root->getInteractionSolver();

    root->varCHorizontal()->setValue(0.5);
    root->varCVertical()->setValue(1.3);
    root->varBouyancyFactor()->setValue(300);
    root->varDragFactor()->setValue(1.0);

    // Sand simulator.
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();  //sandSim->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 0, 20, 5, 25 };
    fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.2f);
    //fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.5f);
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.3);
    //pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    pRenderModule->setColor(Vector3f(1.0f * 0.9, 0.9f * 0.9, 122.0f / 255.0f * 0.9));
    //pRenderModule->setColor(Vector3f(254.0f/255.f, 204.0f/255.f, 153.0f / 255.0f));
    //pRenderModule->setColor(Vector3f(211.0f/255.f, 198.0f/255.f, 166.0f / 255.0f));

    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 3, 2, sandinfo.griddl);

    /// ------  Rigid ------------
    std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    rigidSim->getSolver()->setUseGPU(true);
    rigidSim->needForward(false);
    auto rigidSolver = rigidSim->getSolver();

    root->setRigidSolver(rigidSolver);
    root->addChild(rigidSim);

    {

        double   scale1d = 0.15;
        Vector3f scale(scale1d, scale1d, scale1d);
        double   rhorigid = 2000;
        float    radius   = 1.0;
        radius *= scale1d * 2;
        float    rigid_mass = rhorigid * scale1d * scale1d * scale1d * 8;
        Vector3f rigidI     = RigidUtil::calculateCubeLocalInertia(rigid_mass, scale * 2.0);

        /// rigids body
        auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
        int  id     = rigidSim->addRigid(prigid);

        auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        prigid->addVisualModule(renderModule);
        m_rigids.push_back(prigid);
        m_rigidRenders.push_back(renderModule);

        prigid->setRadius(radius);
        prigid->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
        //triset->translate(Vector3f(0, 0, -0.5));
        triset->scale(scale);

        prigid->setGlobalR(Vector3f(-0.5, 1.2, 0));
        prigid->setGlobalQ(Quaternionf(0, 0, 0, 1).normalize());
        prigid->setExternalForce(Vector3f(0, -5 * rigid_mass, 0));
        prigid->setI(Inertia<float>(rigid_mass, rigidI));

        DistanceField3D<DataType3f> sdf;
        sdf.loadSDF("../../Media/standard/standard_cube.sdf");
        //sdf.translate(Vector3f(0, 0, -0.5) );
        sdf.scale(scale1d);
        interactionSolver->addSDF(sdf);
    }

    {

        double   scale1d = 0.2;
        Vector3f scale(scale1d, scale1d, scale1d);
        double   rhorigid = 2000;
        float    radius   = 1.0;
        radius *= scale1d;
        float    rigid_mass = rhorigid * 4.0 / 3.0 * std::_Pi * radius * radius * radius;
        Vector3f rigidI     = RigidUtil::calculateSphereLocalInertia(rigid_mass, radius);

        /// rigids body
        auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
        int  id     = rigidSim->addRigid(prigid);

        auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        prigid->addVisualModule(renderModule);
        m_rigids.push_back(prigid);
        m_rigidRenders.push_back(renderModule);

        prigid->setRadius(radius);
        prigid->loadShape("../../Media/standard/standard_sphere.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
        //triset->translate(Vector3f(0, 0, -0.5));
        triset->scale(scale);

        prigid->setGlobalR(Vector3f(0.5, 1.2, 0));
        prigid->setGlobalQ(Quaternionf(0, 0, 0, 1).normalize());
        prigid->setExternalForce(Vector3f(0, -5 * rigid_mass, 0));
        prigid->setI(Inertia<float>(rigid_mass, rigidI));

        DistanceField3D<DataType3f> sdf;
        sdf.loadSDF("../../Media/standard/standard_sphere.sdf");
        //sdf.translate(Vector3f(0, 0, -0.5) );
        sdf.scale(scale1d);
        interactionSolver->addSDF(sdf);
    }

    // Add boundary rigid.
    PkAddBoundaryRigid(root, Vector3f(), sandinfo.nx * sandinfo.griddl, sandinfo.ny * sandinfo.griddl, 0.05, 0.15);

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 3, 3.5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0.), Vector3f(0, 1, 0));

    this->disableDisplayFrameRate();
    //this->enableDisplayFrameRate();
    this->enableDisplayFrame();
}

DemoHeightFieldSandLandRigid* DemoHeightFieldSandLandRigid::m_instance = 0;
void                          DemoHeightFieldSandLandRigid::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64 * 2;
    sandinfo.ny               = 32;
    sandinfo.griddl           = 0.05;
    sandinfo.mu               = 0.7;
    sandinfo.drag             = 1;
    sandinfo.slide            = 10 * sandinfo.griddl;
    sandinfo.sandRho          = 1000.0;
    double sandParticleHeight = 0.1;
    double slideAngle         = 15.0 / 180.0 * 3.14159;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(10, 10, 10));
    scene.setLowerBound(Vector3f(-10, -5, -10));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    auto interactionSolver                         = root->getInteractionSolver();
    interactionSolver->m_useStickParticleVelUpdate = false;

    root->varCHorizontal()->setValue(1.5);
    root->varCVertical()->setValue(1.5);
    root->varBouyancyFactor()->setValue(10);
    root->varDragFactor()->setValue(1.0);

    // Sand Simulator.
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();  //sandSim->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 0, 20, 5, 25 };
    //fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.07f);
    //fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.5f);

    // land
    float dhLand = sandinfo.griddl * tan(slideAngle);
    float lhLand = dhLand * sandinfo.nx / 2.0;
    for (int i = 0; i < sandinfo.nx; ++i)
    {
        for (int j = 0; j < sandinfo.ny; ++j)
        {
            //landHeight[j*sandinfo.nx + i] = lhLand - dhLand * i;
            //if (landHeight[j*sandinfo.nx + i] < 0)
            //    landHeight[j*sandinfo.nx + i] = 0.0f;
            double curh = 0;
            if (i < sandinfo.nx / 2.0)
            {
                double r = sandinfo.nx * sandinfo.griddl * 0.8;
                curh     = cos(asin((sandinfo.nx / 2.0 - i) * sandinfo.griddl / r)) * r;
                curh     = r - curh;
            }
            landHeight[j * sandinfo.nx + i] = curh;
        }
    }
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.3);
    pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 3, 2, sandinfo.griddl);
    //psampler->compute();

    /// ------  Rigid ------------
    std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    rigidSim->getSolver()->setUseGPU(true);
    rigidSim->needForward(false);
    auto rigidSolver = rigidSim->getSolver();

    root->setRigidSolver(rigidSolver);
    root->addChild(rigidSim);

    double   scale1d = 0.2;
    Vector3f scale(scale1d, scale1d, scale1d);
    double   rhorigid = 2000;
    float    radius   = 1.0;
    radius *= scale1d;
    float    rigid_mass = rhorigid * 4.0 * std::_Pi * radius * radius * radius;
    Vector3f rigidI     = RigidUtil::calculateSphereLocalInertia(
        rigid_mass, radius);

    {

        /// rigids body
        auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
        int  id     = rigidSim->addRigid(prigid);

        auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        prigid->addVisualModule(renderModule);
        m_rigids.push_back(prigid);
        m_rigidRenders.push_back(renderModule);

        prigid->loadShape("../../Media/standard/standard_sphere.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
        //triset->translate(Vector3f(0, 0, -0.5));
        triset->scale(scale);

        //prigid->setGeometrySize(scale[0], scale[1], scale[2]);
        //prigid->setAngularVelocity(Vector3f(0., 0.0, -1.0));

        prigid->setLinearVelocity(Vector3f(0., 0.0, 0));
        prigid->setGlobalR(Vector3f(-1.5, 0.7, 0));
        prigid->setGlobalQ(Quaternionf(0, 0, 0, 1).normalize());
        prigid->setExternalForce(Vector3f(0, -5 * rigid_mass, 0));
        prigid->setI(Inertia<float>(rigid_mass, rigidI));

        DistanceField3D<DataType3f> sdf;
        sdf.loadSDF("../../Media/standard/standard_sphere.sdf");
        //sdf.translate(Vector3f(0, 0, -0.5) );
        sdf.scale(scale1d);
        interactionSolver->addSDF(sdf);
    }

    // Land mesh.
    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = sandGrid.m_landHeight;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    // Add boundary rigid.
    PkAddBoundaryRigid(root, Vector3f(), sandinfo.nx * sandinfo.griddl, sandinfo.ny * sandinfo.griddl, 0.05, 0.15);

    // Translate camera position
    auto&    camera_ = this->activeCamera();
    Vector3f camPos(0, 3, 3.5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0.), Vector3f(0, 1, 0));

    this->disableDisplayFrameRate();
    //this->enableDisplayFrameRate();
    this->enableDisplayFrame();
}

DemoHeightFieldSandSlide* DemoHeightFieldSandSlide::m_instance = 0;
void                      DemoHeightFieldSandSlide::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64;
    sandinfo.ny               = 64;
    sandinfo.griddl           = 0.05;
    sandinfo.mu               = 0.7;
    sandinfo.drag             = 0.95;
    sandinfo.slide            = 10 * sandinfo.griddl;
    sandinfo.sandRho          = 1000.0;
    double sandParticleHeight = 0.1;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(10, 10, 10));
    scene.setLowerBound(Vector3f(-10, -5, -10));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    auto interactionSolver = root->getInteractionSolver();

    // Sand Simulator.
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();  //sandSim->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 31 - 8, 31 + 8, 13 - 5, 13 + 5 };
    //fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    //fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.2f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 1.f);

    // land
    float lhLand = 0.8;
    float dhLand = lhLand / sandinfo.nx * 2;
    for (int i = 0; i < sandinfo.nx; ++i)
    {
        for (int j = 0; j < sandinfo.ny; ++j)
        {
            landHeight[i * sandinfo.ny + j] = lhLand - dhLand * j;
            if (landHeight[i * sandinfo.ny + j] < 0)
                landHeight[i * sandinfo.ny + j] = 0.0f;
        }
    }
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.5);
    pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 2, 2, sandinfo.griddl);
    //psampler->compute();

    /// ------  Rigid ------------
    std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    rigidSim->getSolver()->setUseGPU(true);
    rigidSim->needForward(false);
    auto rigidSolver = rigidSim->getSolver();

    root->setRigidSolver(rigidSolver);
    root->addChild(rigidSim);

    double   scale1d = 0.15;
    Vector3f scale(scale1d, scale1d, scale1d);
    double   rhorigid = 200000;
    float    radius   = 1.0;
    radius *= scale1d;
    float    rigid_mass = rhorigid * 4.0 * std::_Pi * radius * radius * radius;
    Vector3f rigidI     = RigidUtil::calculateSphereLocalInertia(
        rigid_mass, radius);

    double      rotRad = atan(dhLand / sandinfo.griddl);
    Quaternionf cubeRot(Vector3f(0, 0, -1), rotRad);

    int N = 1;
    for (int i = 0; i < N; ++i)
    {

        /// rigids body
        auto prigid = std::make_shared<RigidBody2<DataType3f>>("rigid");
        int  id     = rigidSim->addRigid(prigid);

        auto renderModule = std::make_shared<RigidMeshRender>(prigid->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        prigid->addVisualModule(renderModule);
        m_rigids.push_back(prigid);
        m_rigidRenders.push_back(renderModule);

        prigid->loadShape("../../Media/standard/standard_cube.obj");
        auto triset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
        //triset->translate(Vector3f(0, 0, -0.5));
        triset->scale(scale);

        //prigid->setGeometrySize(scale[0], scale[1], scale[2]);
        //prigid->setAngularVelocity(Vector3f(0., 0.0, -1.0));

        prigid->setLinearVelocity(Vector3f(0., 0.0, 0));
        prigid->setGlobalR(Vector3f(-0.5 * i - 0.3 /*+ 1.0*/, 0.33 + 0.5 * i, 0));
        prigid->setGlobalQ(/*Quaternionf(0, 0, 0, 1).normalize()*/ cubeRot);
        prigid->setExternalForce(Vector3f(0, -5 * rigid_mass, 0));
        prigid->setI(Inertia<float>(rigid_mass, rigidI));

        DistanceField3D<DataType3f> sdf;
        sdf.loadSDF("../../Media/standard/standard_cube.sdf");
        //sdf.translate(Vector3f(0, 0, -0.5) );
        sdf.scale(scale1d);
        interactionSolver->addSDF(sdf);
    }

    // Land mesh.
    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = sandGrid.m_landHeight;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 1.5, 5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}

DemoHeightFieldSandLandMultiRigid* DemoHeightFieldSandLandMultiRigid::m_instance = 0;
void                               DemoHeightFieldSandLandMultiRigid::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64;
    sandinfo.ny               = 64;
    sandinfo.griddl           = 0.05;
    sandinfo.mu               = 0.7;
    sandinfo.drag             = 0.95;
    sandinfo.slide            = 10 * sandinfo.griddl;
    sandinfo.sandRho          = 1000.0;
    double sandParticleHeight = 0.1;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(10, 10, 10));
    scene.setLowerBound(Vector3f(-10, -5, -10));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    auto interactionSolver = root->getInteractionSolver();

    // Sand simulator.
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();  //sandSim->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 0, 20 + 0, 5, 25 + 0 };
    //fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.05f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.5f);

    // land
    float lhLand = 0.4;
    float dhLand = lhLand / sandinfo.nx * 2;
    for (int i = 0; i < sandinfo.nx; ++i)
    {
        for (int j = 0; j < sandinfo.ny; ++j)
        {
            landHeight[i * sandinfo.ny + j] = lhLand - dhLand * j;
            if (landHeight[i * sandinfo.ny + j] < 0)
                landHeight[i * sandinfo.ny + j] = 0.0f;
        }
    }
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.5);
    pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 2, 2, sandinfo.griddl);

    // Land mesh.
    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = sandGrid.m_landHeight;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    /// ------  Rigid ------------
    std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    rigidSim->getSolver()->setUseGPU(true);
    rigidSim->needForward(false);
    auto rigidSolver = rigidSim->getSolver();

    root->setRigidSolver(rigidSolver);
    root->addChild(rigidSim);

    // Car.
    double   scale1d = 0.2;
    Vector3d scale3d(scale1d, scale1d, scale1d);
    Vector3f scale3f(scale1d, scale1d, scale1d);

    Vector3f chassisCenter;
    Vector3f wheelCenter[4];
    Vector3f chassisSize;
    Vector3f wheelSize[4];

    std::shared_ptr<TriangleSet<DataType3f>> chassisTri;
    std::shared_ptr<TriangleSet<DataType3f>> wheelTri[4];

    DistanceField3D<DataType3f> chassisSDF;
    DistanceField3D<DataType3f> wheelSDF[4];
    // Load car mesh.
    {
        Vector3f boundingsize;
        // Chassis mesh.
        ObjFileLoader chassisLoader("../../Media/car_standard/chassis.obj");

        chassisTri = std::make_shared<TriangleSet<DataType3f>>();
        chassisTri->setPoints(chassisLoader.getVertexList());
        chassisTri->setTriangles(chassisLoader.getFaceList());
        computeBoundingBox(chassisCenter, chassisSize, chassisLoader.getVertexList());
        chassisCenter *= scale3f;
        chassisSize *= scale3f;
        chassisTri->scale(scale3f);
        chassisTri->translate(-chassisCenter);

        // Chassis sdf.
        chassisSDF.loadSDF("../../Media/car_standard/chassis.sdf");
        chassisSDF.scale(scale1d);
        chassisSDF.translate(-chassisCenter);
        //interactionSolver->addSDF(sdf);

        for (int i = 0; i < 4; ++i)
        {
            string objfile("../../Media/car_standard/wheel");
            objfile += std::to_string(i + 1) + ".obj";
            string sdffile("../../Media/car_standard/wheel");
            sdffile += std::to_string(i + 1) + ".sdf";

            // Wheel mesh.
            ObjFileLoader wheelLoader(objfile);
            wheelTri[i] = std::make_shared<TriangleSet<DataType3f>>();
            wheelTri[i]->setPoints(wheelLoader.getVertexList());
            wheelTri[i]->setTriangles(wheelLoader.getFaceList());
            computeBoundingBox(wheelCenter[i], wheelSize[i], wheelLoader.getVertexList());
            wheelCenter[i] *= scale3f;
            wheelSize[i] *= scale3f;
            wheelTri[i]->scale(scale3f);
            wheelTri[i]->translate(-wheelCenter[i]);

            // Wheel sdf.
            DistanceField3D<DataType3f>& sdf = wheelSDF[i];
            sdf.loadSDF(sdffile);
            sdf.scale(scale1d);
            sdf.translate(-wheelCenter[i]);
            //interactionSolver->addSDF(sdf);
        }
    }

    m_car = std::make_shared<PBDCar>();
    rigidSim->addChild(m_car);
    m_car->m_rigidSolver = rigidSolver;

    m_car->carPosition = Vector3f(0.3, 0.3, 0.5) + chassisCenter;
    double rotRad      = 90.0 / 180.0 * std::_Pi;
    m_car->carRotation = Quaternion<float>(-std::sin(rotRad / 2.0), 0, 0., std::cos(rotRad / 2.0)).normalize();
    double rotRad2     = std::_Pi;
    m_car->carRotation = Quaternion<float>(0., std::sin(rotRad2 / 2.0), 0., std::cos(rotRad2 / 2.0)).normalize() * m_car->carRotation;

    m_car->wheelRelPosition[0] = wheelCenter[0] - chassisCenter;
    m_car->wheelRelPosition[1] = wheelCenter[1] - chassisCenter;
    m_car->wheelRelPosition[2] = wheelCenter[2] - chassisCenter;
    m_car->wheelRelPosition[3] = wheelCenter[3] - chassisCenter;
    m_car->wheelRelRotation[0] = Quaternion<float>(0, 0, 0, 1);  // (0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[1] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[2] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[3] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();

    m_car->wheelupDirection    = Vector3f(0, 0, 1);
    m_car->wheelRightDirection = Vector3f(-1, 0, 0);

    m_car->chassisMass    = 1500;  // 00;
    m_car->chassisInertia = RigidUtil::calculateCubeLocalInertia(m_car->chassisMass, chassisSize);

    float wheelm = 50;
    //float wheelRad = wheelTri[0][1]
    Vector3f wheelI        = RigidUtil::calculateCylinderLocalInertia(wheelm,
                                                               (wheelSize[0][1] + wheelSize[0][2]) / 2.0,
                                                               wheelSize[0][0],
                                                               0);
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

    m_car->forwardForceAcc = 1000;
    //m_car->breakForceAcc ;
    m_car->steeringSpeed = 1.0;
    m_car->maxVel        = 2.5;

    // Build.
    m_car->build();

    // Add visualization module and topology module.
    m_car->m_chassis->setTopologyModule(chassisTri);
    auto chassisRender = std::make_shared<RigidMeshRender>(m_car->m_chassis->getTransformationFrame());
    chassisRender->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
    m_car->m_chassis->addVisualModule(chassisRender);
    interactionSolver->addSDF(chassisSDF, m_car->m_chassis->getId());

    // Bounding radius of chassis.
    float chassisRadius = chassisTri->computeBoundingRadius();
    m_car->m_chassis->setRadius(chassisRadius);

    m_rigids.push_back(m_car->m_chassis);
    m_rigidRenders.push_back(chassisRender);

    for (int i = 0; i < 4; ++i)
    {
        m_car->m_wheels[i]->setTopologyModule(wheelTri[i]);
        auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_wheels[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        m_car->m_wheels[i]->addVisualModule(renderModule);
        interactionSolver->addSDF(wheelSDF[i], m_car->m_wheels[i]->getId());

        // Bounding radius of chassis.
        float wheelRadius = wheelTri[i]->computeBoundingRadius();
        m_car->m_wheels[i]->setRadius(wheelRadius);

        m_rigids.push_back(m_car->m_wheels[i]);
        m_rigidRenders.push_back(renderModule);
    }

    interactionSolver->m_prigids = &(rigidSolver->getRigidBodys());

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 1.5, 5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}

DemoHeightFieldSandLandMultiRigid2* DemoHeightFieldSandLandMultiRigid2::m_instance = 0;
void                                DemoHeightFieldSandLandMultiRigid2::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64 * 4;
    sandinfo.ny               = 64 * 4;
    sandinfo.griddl           = 0.03;
    sandinfo.mu               = 0.9;
    sandinfo.drag             = 0.95;
    sandinfo.slide            = 10 * sandinfo.griddl;
    sandinfo.sandRho          = 1000.0;
    double sandParticleHeight = 0.1;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(100, 100, 100));
    scene.setLowerBound(Vector3f(-100, -100, -100));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    auto interactionSolver = root->getInteractionSolver();

    root->varCHorizontal()->setValue(1.);
    root->varCVertical()->setValue(2.);
    root->varBouyancyFactor()->setValue(1);
    root->varDragFactor()->setValue(3.0);

    //
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    m_psandsolver                              = psandSolver;
    psandSolver->setCFLNumber(0.3);
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 0, 20 + 0, 5, 25 + 0 };
    //fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.25f);
    //fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.5f);

    HostHeightField1d hosthf;
    hosthf.resize(sandinfo.nx, sandinfo.ny);
    HeightFieldLoader hfloader;
    double            maxh = 1;
    hfloader.setRange(0, maxh);
    hfloader.load(hosthf, "../../Media/HeightFieldImg/terrain_lying2.png");

    // land
    float lhland = 0.4;
    float dhland = lhland / sandinfo.nx * 2;
    for (int i = 0; i < sandinfo.nx; ++i)
    {
        for (int j = 0; j < sandinfo.ny; ++j)
        {
            //landheight[i*sandinfo.ny + j] = 0;// lhland - dhland * j;
            //if (landheight[i*sandinfo.ny + j] < 0)
            //    landheight[i*sandinfo.ny + j] = 0.0f;
            double curh = 0.5 * maxh - hosthf(i, j);
            if (curh < 0.2)
                curh = 0.2;
            landHeight[i * sandinfo.ny + j] = curh;
        }
    }
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.3);
    //pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    pRenderModule->setColor(Vector3f(1.0f * 0.9, 0.9f * 0.9, 122.0f / 255.0f * 0.9));

    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 3, 2, sandinfo.griddl);

    // Land mesh.
    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = sandGrid.m_landHeight;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    /// ------  Rigid ------------
    std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    rigidSim->getSolver()->setUseGPU(true);
    rigidSim->needForward(false);
    auto rigidSolver = rigidSim->getSolver();

    root->setRigidSolver(rigidSolver);
    root->addChild(rigidSim);

    // Car.
    double   scale1d = 1.;
    Vector3d scale3d(scale1d, scale1d, scale1d);
    Vector3f scale3f(scale1d, scale1d, scale1d);

    Vector3f chassisCenter;
    Vector3f wheelCenter[4];
    Vector3f chassisSize;
    Vector3f wheelSize[4];

    std::shared_ptr<TriangleSet<DataType3f>> chassisTri;
    std::shared_ptr<TriangleSet<DataType3f>> wheelTri[4];

    DistanceField3D<DataType3f> chassisSDF;
    DistanceField3D<DataType3f> wheelSDF[4];
    // Load car mesh.
    {
        Vector3f boundingsize;
        // Chassis mesh.
        ObjFileLoader chassisLoader("../../Media/car2/chassis_cube.obj");

        chassisTri = std::make_shared<TriangleSet<DataType3f>>();
        chassisTri->setPoints(chassisLoader.getVertexList());
        chassisTri->setTriangles(chassisLoader.getFaceList());
        computeBoundingBox(chassisCenter, chassisSize, chassisLoader.getVertexList());
        chassisCenter *= scale3f;
        chassisSize *= scale3f;
        chassisTri->scale(scale3f);
        chassisTri->translate(-chassisCenter);

        // Chassis sdf.
        chassisSDF.loadSDF("../../Media/car2/chassis_cube.sdf");
        chassisSDF.scale(scale1d);
        chassisSDF.translate(-chassisCenter);
        //interactionSolver->addSDF(sdf);

        for (int i = 0; i < 4; ++i)
        {
            string objfile("../../Media/car2/wheel.obj");
            string sdffile("../../Media/car2/wheel.sdf");

            // Wheel mesh.
            ObjFileLoader wheelLoader(objfile);
            wheelTri[i] = std::make_shared<TriangleSet<DataType3f>>();
            wheelTri[i]->setPoints(wheelLoader.getVertexList());
            wheelTri[i]->setTriangles(wheelLoader.getFaceList());
            computeBoundingBox(wheelCenter[i], wheelSize[i], wheelLoader.getVertexList());
            wheelCenter[i] *= scale3f;
            wheelSize[i] *= scale3f;
            wheelTri[i]->scale(scale3f);
            wheelTri[i]->translate(-wheelCenter[i]);

            // Wheel sdf.
            DistanceField3D<DataType3f>& sdf = wheelSDF[i];
            sdf.loadSDF(sdffile);
            sdf.scale(scale1d);
            sdf.translate(-wheelCenter[i]);
            //interactionSolver->addSDF(sdf);
        }
    }

    m_car = std::make_shared<PBDCar>();
    rigidSim->addChild(m_car);
    m_car->m_rigidSolver = rigidSolver;

    m_car->carPosition = Vector3f(0.35, 0.65, 1.5) + chassisCenter;
    //double rotRad = 90.0 / 180.0 * std::_Pi;
    //m_car->carRotation = Quaternion<float>(-std::sin(rotRad / 2.0), 0, 0., std::cos(rotRad / 2.0)).normalize();
    //double rotRad2 = std::_Pi;
    //m_car->carRotation = Quaternion<float>(0., std::sin(rotRad2 / 2.0), 0., std::cos(rotRad2 / 2.0)).normalize() * m_car->carRotation;

    m_car->wheelRelPosition[0] = Vector3f(-0.3f, -0.2, -0.4f /* -0.01*/) * scale1d + wheelCenter[0] - chassisCenter;
    m_car->wheelRelPosition[1] = Vector3f(+0.3f /*+0.01*/, -0.2, -0.4f /* +0.02*/) * scale1d + wheelCenter[1] - chassisCenter;
    m_car->wheelRelPosition[2] = Vector3f(-0.3f, -0.2, 0.4f) * scale1d + wheelCenter[2] - chassisCenter;
    m_car->wheelRelPosition[3] = Vector3f(+0.3f, -0.2, 0.4f) * scale1d + wheelCenter[3] - chassisCenter;
    m_car->wheelRelRotation[0] = Quaternion<float>(0, 0, 0, 1);  // (0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[1] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[2] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();
    m_car->wheelRelRotation[3] = Quaternion<float>(0, 0, 0, 1);  //(0, 0.5, 0, 0.5).normalize();

    m_car->wheelupDirection    = Vector3f(0, 1, 0);
    m_car->wheelRightDirection = Vector3f(1, 0, 0);

    m_car->chassisMass    = 5000;  // 00;
    m_car->chassisInertia = RigidUtil::calculateCubeLocalInertia(m_car->chassisMass, chassisSize);

    float wheelm = 150;
    //float wheelRad = wheelTri[0][1]
    Vector3f wheelI        = RigidUtil::calculateCylinderLocalInertia(wheelm,
                                                               (wheelSize[0][1] + wheelSize[0][2]) / 2.0,
                                                               wheelSize[0][0],
                                                               0);
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

    m_car->forwardForceAcc = 1000;
    //m_car->breakForceAcc ;
    m_car->steeringSpeed = 1.0;
    m_car->maxVel        = 2.5;

    // Build.
    m_car->build();

    // Add visualization module and topology module.
    m_car->m_chassis->setTopologyModule(chassisTri);
    auto chassisRender = std::make_shared<RigidMeshRender>(m_car->m_chassis->getTransformationFrame());
    chassisRender->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
    m_car->m_chassis->addVisualModule(chassisRender);
    interactionSolver->addSDF(chassisSDF, m_car->m_chassis->getId());

    // Bounding radius of chassis.
    float chassisRadius = chassisTri->computeBoundingRadius();
    m_car->m_chassis->setRadius(chassisRadius);

    m_rigids.push_back(m_car->m_chassis);
    m_rigidRenders.push_back(chassisRender);

    for (int i = 0; i < 4; ++i)
    {
        m_car->m_wheels[i]->setTopologyModule(wheelTri[i]);
        auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_wheels[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        m_car->m_wheels[i]->addVisualModule(renderModule);
        interactionSolver->addSDF(wheelSDF[i], m_car->m_wheels[i]->getId());

        // Bounding radius of chassis.
        float wheelRadius = wheelTri[i]->computeBoundingRadius();
        m_car->m_wheels[i]->setRadius(wheelRadius);

        m_rigids.push_back(m_car->m_wheels[i]);
        m_rigidRenders.push_back(renderModule);
    }

    interactionSolver->m_prigids = &(rigidSolver->getRigidBodys());

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 3, 5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}

void DemoHeightFieldSandLandMultiRigid2::_setSandHeightTo(float h)
{
    if (!m_psandsolver)
        return;

    SandGridInfo*     sandinfo = m_psandsolver->getSandGridInfo();
    HostHeightField1d sandheight;
    sandheight.resize(sandinfo->nx, sandinfo->ny);

    for (int i = 0; i < sandinfo->nx; ++i)
    {
        for (int j = 0; j < sandinfo->ny; ++j)
        {
            sandheight(i, j) = h;
        }
    }
    m_psandsolver->setSandGridHeight(sandheight);

    sandheight.Release();
}

DemoHeightFieldSandLandMultiRigidTest* DemoHeightFieldSandLandMultiRigidTest::m_instance = 0;
void                                   DemoHeightFieldSandLandMultiRigidTest::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx               = 64;
    sandinfo.ny               = 64;
    sandinfo.griddl           = 0.05;
    sandinfo.mu               = 0.7;
    sandinfo.drag             = 0.95;
    sandinfo.slide            = 10 * sandinfo.griddl;
    sandinfo.sandRho          = 1000.0;
    double sandParticleHeight = 0.1;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(10, 10, 10));
    scene.setLowerBound(Vector3f(-10, -5, -10));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    auto interactionSolver = root->getInteractionSolver();

    //
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    std::vector<int>   humpBlock = { 0, 20, 5, 25 };
    //fillGrid2D(&(landHeight[0]), sandinfo.nx, sandinfo.ny, 0.0f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, 0.2f);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.5f);

    // land
    float lhLand = 0.4;
    float dhLand = lhLand / sandinfo.nx * 2;
    for (int i = 0; i < sandinfo.nx; ++i)
    {
        for (int j = 0; j < sandinfo.ny; ++j)
        {
            landHeight[i * sandinfo.ny + j] = lhLand - dhLand * j;
            if (landHeight[i * sandinfo.ny + j] < 0)
                landHeight[i * sandinfo.ny + j] = 0.0f;
        }
    }
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    //pRenderModule->varRenderMode()->getValue().currentKey() = PointRenderModule::RenderModeEnum::SPRITE;
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.5);
    pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 2, 2, sandinfo.griddl);

    // Land mesh.
    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = sandGrid.m_landHeight;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    /// ------  Rigid ------------
    std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    rigidSim->getSolver()->setUseGPU(true);
    rigidSim->needForward(false);
    auto rigidSolver = rigidSim->getSolver();

    root->setRigidSolver(rigidSolver);
    root->addChild(rigidSim);

    // Car.
    double   scale1d = 0.2;
    Vector3d scale3d(scale1d, scale1d, scale1d);
    Vector3f scale3f(scale1d, scale1d, scale1d);

    Vector3f chassisCenter;
    Vector3f wheelCenter[4];
    Vector3f chassisSize;
    Vector3f wheelSize[4];

    std::shared_ptr<TriangleSet<DataType3f>> chassisTri;
    std::shared_ptr<TriangleSet<DataType3f>> wheelTri[4];

    DistanceField3D<DataType3f> chassisSDF;
    DistanceField3D<DataType3f> wheelSDF[4];
    // Load car mesh.
    {
        Vector3f boundingsize;
        // Chassis mesh.
        ObjFileLoader chassisLoader("../../Media/car_standard/chassis.obj");
        chassisTri = std::make_shared<TriangleSet<DataType3f>>();
        chassisTri->setPoints(chassisLoader.getVertexList());
        chassisTri->setTriangles(chassisLoader.getFaceList());
        computeBoundingBox(chassisCenter, chassisSize, chassisLoader.getVertexList());
        chassisCenter *= scale3f;
        chassisSize *= scale3f;
        chassisTri->scale(scale3f);
        chassisTri->translate(-chassisCenter);

        // Chassis sdf.
        chassisSDF.loadSDF("../../Media/car_standard/chassis.sdf");
        chassisSDF.scale(scale1d);
        chassisSDF.translate(-chassisCenter);
        //interactionSolver->addSDF(sdf);

        for (int i = 0; i < 4; ++i)
        {
            string objfile("../../Media/car_standard/wheel");
            objfile += std::to_string(i + 1) + ".obj";
            string sdffile("../../Media/car_standard/wheel");
            sdffile += std::to_string(i + 1) + ".sdf";

            // Wheel mesh.
            ObjFileLoader wheelLoader(objfile);
            wheelTri[i] = std::make_shared<TriangleSet<DataType3f>>();
            wheelTri[i]->setPoints(wheelLoader.getVertexList());
            wheelTri[i]->setTriangles(wheelLoader.getFaceList());
            computeBoundingBox(wheelCenter[i], wheelSize[i], wheelLoader.getVertexList());
            wheelCenter[i] *= scale3f;
            wheelSize[i] *= scale3f;
            wheelTri[i]->scale(scale3f);
            wheelTri[i]->translate(-wheelCenter[i]);

            // Wheel sdf.
            DistanceField3D<DataType3f>& sdf = wheelSDF[i];
            sdf.loadSDF(sdffile);
            sdf.scale(scale1d);
            sdf.translate(-wheelCenter[i]);
            //interactionSolver->addSDF(sdf);
        }
    }

    RigidBody2_ptr chassis;
    RigidBody2_ptr wheel[4];

    float wheelm = 50;
    //float wheelRad = wheelTri[0][1]
    Vector3f wheelI         = RigidUtil::calculateCylinderLocalInertia(wheelm,
                                                               (wheelSize[0][1] + wheelSize[0][2]) / 2.0,
                                                               wheelSize[0][0],
                                                               0);
    float    chassisMass    = 1500;  // 00;
    Vector3f chassisInertia = RigidUtil::calculateCubeLocalInertia(chassisMass, chassisSize);

    double            rotRad = 90.0 / 180.0 * std::_Pi;
    Quaternion<float> q1(-std::sin(rotRad / 2.0), 0, 0., std::cos(rotRad / 2.0));
    double            rotRad2 = std::_Pi;
    Quaternionf       q2      = Quaternion<float>(0., std::sin(rotRad2 / 2.0), 0., std::cos(rotRad2 / 2.0)).normalize() * q1.normalize();

    //chassis = std::make_shared<RigidBody2<DataType3f>>();
    //rigidSim->addRigid(chassis);

    //chassis->setI(Inertia<float>(chassisMass, chassisInertia));
    //chassis->setGlobalR(Vector3f(0, 0.7, 0));

    //chassis->setGlobalQ(q2);
    //chassis->setExternalForce(Vector3f(0.0*chassisMass, -9.8 * chassisMass, 0));

    ////int idchassis = rigidSolver->addRigid(chassis);

    //// Add visualization module and topology module.
    //chassis->setTopologyModule(chassisTri);
    //auto chassisRender = std::make_shared<RigidMeshRender>(chassis->getTransformationFrame());
    //chassisRender->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
    //chassis->addVisualModule(chassisRender);
    //interactionSolver->addSDF(chassisSDF, chassis->getId());

    //m_rigids.push_back(chassis);
    //m_rigidRenders.push_back(chassisRender);

    for (int i = 0; i < 4; ++i)
    {

        wheel[i] = std::make_shared<RigidBody2<DataType3f>>();

        wheel[i]->setI(Inertia<float>(wheelm, wheelI));
        wheel[i]->setGlobalR(Vector3f(0, 0.7, 0) + q2.rotate(wheelCenter[i] - chassisCenter));
        wheel[i]->setGlobalQ(q2);
        wheel[i]->setExternalForce(Vector3f(0.0 * wheelm, -9.8 * wheelm, 0));

        rigidSim->addRigid(wheel[i]);

        wheel[i]->setTopologyModule(wheelTri[i]);
        auto renderModule = std::make_shared<RigidMeshRender>(wheel[i]->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double) 1000, 0.8));
        wheel[i]->addVisualModule(renderModule);
        interactionSolver->addSDF(wheelSDF[i], wheel[i]->getId());

        m_rigids.push_back(wheel[i]);
        m_rigidRenders.push_back(renderModule);
    }

    interactionSolver->m_prigids = &(rigidSolver->getRigidBodys());

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 1.5, 5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}

DemoHeightFieldSandValley* DemoHeightFieldSandValley::m_instance = 0;
void                       DemoHeightFieldSandValley::createScene()
{
    SandGridInfo sandinfo;
    sandinfo.nx      = 512;  // 64 * 5 * 2;
    sandinfo.ny      = 512;  // 64 * 5 * 2;
    sandinfo.griddl  = 0.01 / 2.0;
    sandinfo.mu      = 0.4;
    sandinfo.drag    = 0.98;
    sandinfo.slide   = 10 * sandinfo.griddl;
    sandinfo.sandRho = 1000.0;
    //double sandParticleHeight = 0.1;

    float tanSlope = 0.5;

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(100, 100, 100));
    scene.setLowerBound(Vector3f(-100, -100, -100));

    // Root node. Also the simulator.
    std::shared_ptr<HeightFieldSandRigidInteraction> root = scene.createNewScene<HeightFieldSandRigidInteraction>();
    root->setActive(true);
    root->setDt(0.02);
    auto interactionSolver = root->getInteractionSolver();

    root->varCHorizontal()->setValue(1.);
    root->varCVertical()->setValue(2.);
    root->varBouyancyFactor()->setValue(1);
    root->varDragFactor()->setValue(3.0);

    //
    std::shared_ptr<SandSimulator> sandSim     = std::make_shared<SandSimulator>();
    std::shared_ptr<SSESandSolver> psandSolver = std::make_shared<SSESandSolver>();
    m_psandsolver                              = psandSolver;
    psandSolver->setCFLNumber(0.1);
    sandSim->needForward(false);
    sandSim->setSandSolver(psandSolver);
    root->setSandSolver(psandSolver);
    root->addChild(sandSim);

    // Initialize sand grid data.
    SandGrid& sandGrid = psandSolver->getSandGrid();
    sandGrid.setSandInfo(sandinfo);
    root->setSandGrid(sandGrid.m_sandHeight, sandGrid.m_landHeight);

    float            normalizeC[2] = { 0.1, 0.68 };
    float            normalizeS[2] = { 0.04, 0.03 };
    std::vector<int> humpBlock(4);

    humpBlock[0] = sandinfo.nx * (normalizeC[0] - normalizeS[0]);
    humpBlock[1] = sandinfo.nx * (normalizeC[0] + normalizeS[0]);
    humpBlock[2] = sandinfo.ny * (normalizeC[1] - normalizeS[1]);
    humpBlock[3] = sandinfo.ny * (normalizeC[1] + normalizeS[1]);

    // Height
    std::vector<float> landHeight(sandinfo.nx * sandinfo.ny);
    std::vector<float> surfaceHeight(sandinfo.nx * sandinfo.ny);
    fillGrid2D(&(surfaceHeight[0]), sandinfo.nx, sandinfo.ny, humpBlock, 0.1f);

    HostHeightField1d hosthf;
    hosthf.resize(sandinfo.nx, sandinfo.ny);
    HeightFieldLoader hfloader;
    double            maxh = 0.3;
    hfloader.setRange(0, maxh);
    hfloader.load(hosthf, "../../Media/HeightFieldImg/valley2.png");

    // land
    float lhland = 0.4;
    float dhland = lhland / sandinfo.nx * 2;
    for (int i = 0; i < sandinfo.nx; ++i)
    {
        for (int j = 0; j < sandinfo.ny; ++j)
        {
            float landh                     = hosthf(i, j) - ((i - hosthf.Nx() / 2) * sandinfo.griddl * tanSlope - 1);
            landHeight[j * sandinfo.nx + i] = landh;

            surfaceHeight[j * sandinfo.nx + i] += landh;
        }
    }
    sandGrid.initialize(&(landHeight[0]), &(surfaceHeight[0]));

    // Rendering module of simulator.
    auto pRenderModule = std::make_shared<PointRenderModule>();
    pRenderModule->setSphereInstaceSize(sandinfo.griddl * 0.3);
    pRenderModule->setColor(Vector3f(1.0f, 1.0f, 122.0f / 255.0f));
    //pRenderModule->setColor(Vector3f(1.0f*0.9, 0.9f*0.9, 122.0f / 255.0f*0.9));

    sandSim->addVisualModule(pRenderModule);

    // topology
    auto topology = std::make_shared<PointSet<DataType3f>>();
    sandSim->setTopologyModule(topology);
    topology->getPoints().resize(1);

    // Render point sampler (module).
    auto psampler = std::make_shared<SandHeightRenderParticleSampler>();
    sandSim->addCustomModule(psampler);
    psampler->m_sandHeight = &sandGrid.m_sandHeight;
    psampler->m_landHeight = &sandGrid.m_landHeight;
    psampler->Initalize(sandinfo.nx, sandinfo.ny, 3, 2, sandinfo.griddl);

    // Land mesh.
    {
        auto landrigid = std::make_shared<RigidBody2<DataType3f>>("Land");
        root->addChild(landrigid);

        // Mesh triangles.
        auto triset = std::make_shared<TriangleSet<DataType3f>>();
        landrigid->setTopologyModule(triset);

        // Generate mesh.
        auto&           hfland = sandGrid.m_landHeight;
        HeightFieldMesh hfmesh;
        hfmesh.generate(triset, hfland);

        // Mesh renderer.
        auto renderModule = std::make_shared<RigidMeshRender>(landrigid->getTransformationFrame());
        renderModule->setColor(Vector3f(210.0 / 255.0, 180.0 / 255.0, 140.0 / 255.0));
        landrigid->addVisualModule(renderModule);
    }

    ///// ------  Rigid ------------
    //std::shared_ptr<PBDSolverNode> rigidSim = std::make_shared<PBDSolverNode>();
    //rigidSim->getSolver()->setUseGPU(true);
    //rigidSim->needForward(false);
    //auto rigidSolver = rigidSim->getSolver();

    //root->setRigidSolver(rigidSolver);
    //root->addChild(rigidSim);

    //// Car.
    //double scale1d = 1.;
    //Vector3d scale3d(scale1d, scale1d, scale1d);
    //Vector3f scale3f(scale1d, scale1d, scale1d);

    //Vector3f chassisCenter;
    //Vector3f wheelCenter[4];
    //Vector3f chassisSize;
    //Vector3f wheelSize[4];

    //std::shared_ptr< TriangleSet<DataType3f>> chassisTri;
    //std::shared_ptr< TriangleSet<DataType3f>> wheelTri[4];

    //DistanceField3D<DataType3f> chassisSDF;
    //DistanceField3D<DataType3f> wheelSDF[4];
    //// Load car mesh.
    //{
    //    Vector3f boundingsize;
    //    // Chassis mesh.
    //    ObjFileLoader chassisLoader("../../Media/car2/chassis_cube.obj");

    //    chassisTri = std::make_shared<TriangleSet<DataType3f>>();
    //    chassisTri->setPoints(chassisLoader.getVertexList());
    //    chassisTri->setTriangles(chassisLoader.getFaceList());
    //    computeBoundingBox(chassisCenter, chassisSize, chassisLoader.getVertexList());
    //    chassisCenter *= scale3f;    chassisSize *= scale3f;
    //    chassisTri->scale(scale3f);
    //    chassisTri->translate(-chassisCenter);

    //    // Chassis sdf.
    //    chassisSDF.loadSDF("../../Media/car2/chassis_cube.sdf");
    //    chassisSDF.scale(scale1d);
    //    chassisSDF.translate(-chassisCenter);
    //    //interactionSolver->addSDF(sdf);

    //    for (int i = 0; i < 4; ++i)
    //    {
    //        string objfile("../../Media/car2/wheel.obj");
    //        string sdffile("../../Media/car2/wheel.sdf");

    //        // Wheel mesh.
    //        ObjFileLoader wheelLoader(objfile);
    //        wheelTri[i] = std::make_shared<TriangleSet<DataType3f>>();
    //        wheelTri[i]->setPoints(wheelLoader.getVertexList());
    //        wheelTri[i]->setTriangles(wheelLoader.getFaceList());
    //        computeBoundingBox(wheelCenter[i], wheelSize[i], wheelLoader.getVertexList());
    //        wheelCenter[i] *= scale3f;    wheelSize[i] *= scale3f;
    //        wheelTri[i]->scale(scale3f);
    //        wheelTri[i]->translate(-wheelCenter[i]);

    //        // Wheel sdf.
    //        DistanceField3D<DataType3f>& sdf = wheelSDF[i];
    //        sdf.loadSDF(sdffile);
    //        sdf.scale(scale1d);
    //        sdf.translate(-wheelCenter[i]);
    //        //interactionSolver->addSDF(sdf);
    //    }
    //}

    //m_car = std::make_shared<PBDCar>();
    //rigidSim->addChild(m_car);
    //m_car->m_rigidSolver = rigidSolver;

    //m_car->carPosition = Vector3f(0.35, 0.65, 1.5) + chassisCenter;
    ////double rotRad = 90.0 / 180.0 * std::_Pi;
    ////m_car->carRotation = Quaternion<float>(-std::sin(rotRad / 2.0), 0, 0., std::cos(rotRad / 2.0)).normalize();
    ////double rotRad2 = std::_Pi;
    ////m_car->carRotation = Quaternion<float>(0., std::sin(rotRad2 / 2.0), 0., std::cos(rotRad2 / 2.0)).normalize() * m_car->carRotation;

    //m_car->wheelRelPosition[0] = Vector3f(-0.3f, -0.2, -0.4f/* -0.01*/)*scale1d + wheelCenter[0] - chassisCenter;
    //m_car->wheelRelPosition[1] = Vector3f(+0.3f/*+0.01*/, -0.2, -0.4f/* +0.02*/)*scale1d + wheelCenter[1] - chassisCenter;
    //m_car->wheelRelPosition[2] = Vector3f(-0.3f, -0.2, 0.4f)*scale1d + wheelCenter[2] - chassisCenter;
    //m_car->wheelRelPosition[3] = Vector3f(+0.3f, -0.2, 0.4f)*scale1d + wheelCenter[3] - chassisCenter;
    //m_car->wheelRelRotation[0] = Quaternion<float>(0, 0, 0, 1);// (0, 0.5, 0, 0.5).normalize();
    //m_car->wheelRelRotation[1] = Quaternion<float>(0, 0, 0, 1);//(0, 0.5, 0, 0.5).normalize();
    //m_car->wheelRelRotation[2] = Quaternion<float>(0, 0, 0, 1);//(0, 0.5, 0, 0.5).normalize();
    //m_car->wheelRelRotation[3] = Quaternion<float>(0, 0, 0, 1);//(0, 0.5, 0, 0.5).normalize();

    //m_car->wheelupDirection = Vector3f(0, 1, 0);
    //m_car->wheelRightDirection = Vector3f(1, 0, 0);

    //m_car->chassisMass = 5000;// 00;
    //m_car->chassisInertia = RigidUtil::calculateCubeLocalInertia(m_car->chassisMass, chassisSize);

    //float wheelm = 150;
    ////float wheelRad = wheelTri[0][1]
    //Vector3f wheelI = RigidUtil::calculateCylinderLocalInertia(wheelm,
    //    (wheelSize[0][1] + wheelSize[0][2]) / 2.0, wheelSize[0][0], 0);
    //m_car->wheelMass[0] = wheelm;
    //m_car->wheelInertia[0] = wheelI;
    //m_car->wheelMass[1] = wheelm;
    //m_car->wheelInertia[1] = wheelI;
    //m_car->wheelMass[2] = wheelm;
    //m_car->wheelInertia[2] = wheelI;
    //m_car->wheelMass[3] = wheelm;
    //m_car->wheelInertia[3] = wheelI;

    //m_car->steeringLowerBound = -0.5;
    //m_car->steeringUpperBound = 0.5;

    //m_car->forwardForceAcc = 1000;
    ////m_car->breakForceAcc ;
    //m_car->steeringSpeed = 1.0;
    //m_car->maxVel = 2.5;

    //// Build.
    //m_car->build();

    //// Add visualization module and topology module.
    //m_car->m_chassis->setTopologyModule(chassisTri);
    //auto chassisRender = std::make_shared<RigidMeshRender>(m_car->m_chassis->getTransformationFrame());
    //chassisRender->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
    //m_car->m_chassis->addVisualModule(chassisRender);
    //interactionSolver->addSDF(chassisSDF, m_car->m_chassis->getId());

    //// Bounding radius of chassis.
    //float chassisRadius = chassisTri->computeBoundingRadius();
    //m_car->m_chassis->setRadius(chassisRadius);

    //m_rigids.push_back(m_car->m_chassis);
    //m_rigidRenders.push_back(chassisRender);

    //for (int i = 0; i < 4; ++i)
    //{
    //    m_car->m_wheels[i]->setTopologyModule(wheelTri[i]);
    //    auto renderModule = std::make_shared<RigidMeshRender>(m_car->m_wheels[i]->getTransformationFrame());
    //    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
    //    m_car->m_wheels[i]->addVisualModule(renderModule);
    //    interactionSolver->addSDF(wheelSDF[i], m_car->m_wheels[i]->getId());

    //    // Bounding radius of chassis.
    //    float wheelRadius = wheelTri[i]->computeBoundingRadius();
    //    m_car->m_wheels[i]->setRadius(wheelRadius);

    //    m_rigids.push_back(m_car->m_wheels[i]);
    //    m_rigidRenders.push_back(renderModule);
    //}

    //interactionSolver->m_prigids = &(rigidSolver->getRigidBodys());

    this->disableDisplayFrameRate();

    // Translate camera position
    auto& camera_ = this->activeCamera();
    //camera_.translate(Vector3f(0, 1.5, 3));
    //camera_.setEyePostion(Vector3f(1.5, 1.5, 6));
    Vector3f camPos(0, 3, 5);
    camera_.lookAt(camPos, Vector3f(0, 0, 0), Vector3f(0, 1, 0));
}

void DemoHeightFieldSandValley::_setSandHeightTo(float h)
{
    if (!m_psandsolver)
        return;

    SandGridInfo*     sandinfo = m_psandsolver->getSandGridInfo();
    HostHeightField1d sandheight;
    sandheight.resize(sandinfo->nx, sandinfo->ny);

    for (int i = 0; i < sandinfo->nx; ++i)
    {
        for (int j = 0; j < sandinfo->ny; ++j)
        {
            sandheight(i, j) = h;
        }
    }
    m_psandsolver->setSandGridHeight(sandheight);

    sandheight.Release();
}
