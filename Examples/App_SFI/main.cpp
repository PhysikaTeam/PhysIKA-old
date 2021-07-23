/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2021-06-06
 * @description: Simulate coupling between PBD fluid and projective-peridynamics elastic bodies
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/SolidFluidInteraction.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

/**
 * setup scene: dam-break with 3 elastic bunnies
 * fluid and sfi is modeled with PBD
 * elastic object is modeled with projective-peridynamics
 */
void createScene()
{
    SceneGraph&                                 scene = SceneGraph::getInstance();
    std::shared_ptr<StaticBoundary<DataType3f>> root  = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.015f, true);  //scene boundary

    //fluid
    std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
    root->addParticleSystem(fluid);
    fluid->loadParticles(Vector3f(0), Vector3f(0.5, 1.0, 1.0), 0.015f);
    fluid->setMass(10);
    std::shared_ptr<PositionBasedFluidModel<DataType3f>> pbd = std::make_shared<PositionBasedFluidModel<DataType3f>>();
    fluid->setNumericalModel(pbd);
    fluid->currentPosition()->connect(&pbd->m_position);
    fluid->currentVelocity()->connect(&pbd->m_velocity);
    fluid->currentForce()->connect(&pbd->m_forceDensity);
    pbd->setSmoothingLength(0.02);

    auto ptRender = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(0, 0, 1));
    ptRender->setColorRange(0, 1);
    fluid->addVisualModule(ptRender);
    //pbd-based SFI
    std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
    sfi->setInteractionDistance(0.02);
    root->addChild(sfi);
    sfi->addParticleSystem(fluid);  //feed fluid data to SFI

    //elastic bodies
    int obj_num = 3;
    for (int i = 0; i < obj_num; i++)
    {
        std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
        root->addParticleSystem(bunny);
        bunny->setMass(1.0);
        bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
        bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
        bunny->translate(Vector3f(0.75, 0.2, 0.4 + i * 0.3));
        bunny->setVisible(false);
        bunny->getElasticitySolver()->setIterationNumber(10);
        bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
        bunny->getTopologyMapping()->setSearchingRadius(0.05);
        auto sRender = std::make_shared<SurfaceMeshRender>();
        bunny->getSurfaceNode()->addVisualModule(sRender);
        sRender->setColor(Vector3f(i * 0.3f, 1 - i * 0.3f, 1.0));

        sfi->addParticleSystem(bunny);  //feed elastic object data to SFI
    }
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
