/**
 * @author     : Yue Chang (yuechang@pku.edu.cn)
 * @date       : 2021-07-23
 * @description: Simulate the coupling between PBD fluid and projective-peridynamics elastic bodies with a tube-shaped boundary
 *               reference <Position Based Fluids>
 * @version    : 1.0
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ParticleFluidFast.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "GUI/GlutGUI/GLApp.h"

#include "SFIFast.h"

using namespace std;
using namespace PhysIKA;

/**
 * setup scene: dambreak with elastic objects and tube-shaped boundary
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(4.0f, 1.0f, 1.0f));
    scene.setLowerBound(Vector3f(-0.0f, 0, -0.0f));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(-0, 0, 0), Vector3f(4.0f, 1.0f, 1.0f), 0.025f, true);  //scene boundary
    //fluid
    std::shared_ptr<ParticleFluidFast<DataType3f>> fluid = std::make_shared<ParticleFluidFast<DataType3f>>();
    root->addParticleSystem(fluid);
    fluid->loadParticles("../../Media/fluid/data_fluid_pos.obj");
    fluid->setMass(10);
    fluid->setActive(true);
    fluid->self_update = true;
    auto ptRender      = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(1, 0, 0));
    ptRender->setColorRange(0, 4);
    fluid->addVisualModule(ptRender);
    fluid->currentVelocity()->connect(&ptRender->m_vecIndex);
    //elastic bunny

    std::shared_ptr<SFIFast<DataType3f>> sfi = std::make_shared<SFIFast<DataType3f>>();
    sfi->addParticleSystem(fluid);

    int obj_num = 3;
    for (int i = 0; i < obj_num; i++)
        for (int j = 0; j < obj_num; j++)
        {
            std::shared_ptr<ParticleElasticBody<DataType3f>> cube = std::make_shared<ParticleElasticBody<DataType3f>>();
            root->addParticleSystem(cube);
            cube->setMass(2);
            cube->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
            cube->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
            cube->scale(1.0f / 3.0f);
            cube->translate(Vector3f(0.17 + 0.28 * j, 0.425, 0.2 + 0.3 * i));
            cube->setVisible(false);

            auto sRender = std::make_shared<SurfaceMeshRender>();
            cube->getSurfaceNode()->addVisualModule(sRender);
            sRender->setColor(Vector3f(1.0, 1.0, 1.0));
            //sfi
            sfi->addParticleSystem(cube);
        }

    //int obj_num = 3;
    for (int i = 0; i < obj_num; i++)
        for (int j = 0; j < obj_num; j++)
        {
            std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
            root->addParticleSystem(bunny);
            //root->addParticleSystem(bunny);
            bunny->setMass(2);
            bunny->loadParticles(Vector3f(0), Vector3f(1.0, 1.0, 1.0), 0.1f);
            bunny->loadSurface("../../Media/standard/standard_cube_01.obj");
            bunny->scale(0.05f);
            bunny->translate(Vector3f(0.3 + 0.27 * j, 0.4, 0.3 + 0.32 * i));
            bunny->setVisible(false);

            auto sRender = std::make_shared<SurfaceMeshRender>();
            bunny->getSurfaceNode()->addVisualModule(sRender);
            sRender->setColor(Vector3f(0.0, 1.0, 1.0));
            //sfi
            sfi->addParticleSystem(bunny);
        }
    root->addChild(sfi);
}

int main()
{
    createScene();

    Log::setOutput("console_log.txt");
    Log::setLevel(Log::Info);
    Log::sendMessage(Log::Info, "Simulation begin");

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();

    Log::sendMessage(Log::Info, "Simulation end!");
    return 0;
}
