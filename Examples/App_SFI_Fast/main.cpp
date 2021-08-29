/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2021-06-17
 * @description: Simulate Solid Fluid Interaction with fast solver
 *               reference <Position Based Fluids>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-17
 * @description: poslish code
 * @version    : 1.1
 * @TODO       : find out how fast solver works
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
 * setup scene: dambreak with an elastic bunny
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.015f, true);  //scene boundary
    //fluid
    std::shared_ptr<ParticleFluidFast<DataType3f>> fluid = std::make_shared<ParticleFluidFast<DataType3f>>();
    root->addParticleSystem(fluid);
    fluid->loadParticles(Vector3f(0), Vector3f(0.5, 0.5, 0.5), 0.005f);
    fluid->setMass(10);
    fluid->setActive(true);
    fluid->self_update = false;
    auto ptRender      = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(1, 0, 0));
    ptRender->setColorRange(0, 4);
    fluid->addVisualModule(ptRender);
    fluid->currentVelocity()->connect(&ptRender->m_vecIndex);
    //elastic bunny
    std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
    root->addParticleSystem(bunny);
    bunny->setMass(1.0);
    bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
    bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
    bunny->scale(1.0f);
    bunny->translate(Vector3f(0.75, 0.2, 0.4 + 0.3));
    bunny->setVisible(false);
    bunny->getElasticitySolver()->setIterationNumber(10);
    bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
    bunny->getTopologyMapping()->setSearchingRadius(0.05);
    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(0.0, 1.0, 1.0));
    //sfi
    std::shared_ptr<SFIFast<DataType3f>> sfi = std::make_shared<SFIFast<DataType3f>>();
    sfi->addParticleSystem(fluid);
    sfi->addParticleSystem(bunny);
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
