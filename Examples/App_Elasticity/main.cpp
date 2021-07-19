
/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-02
 * @description: Simulate elastic body with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 */

#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "GUI/GlutGUI/GLApp.h"

#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"

using namespace PhysIKA;

/**
 * A simple drop scene: elastic body falls to the ground
 */
int main()
{
    Log::sendMessage(Log::Info, "Simulation start");

    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);

    std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
    root->addParticleSystem(bunny);

    bunny->setMass(1.0);
    bunny->loadParticles("../../Media/bunny/bunny_points.obj");
    bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");

    // Issues with following models:
    // 1. particles collapse
    // 2. surface mesh explodes
    // TODO: find out why
    // bunny->loadParticles("../../Media/dragon/dragon_points_1190.obj");
    // bunny->loadSurface("../../Media/dragon/dragon.obj");
    // bunny->loadParticles("../../Media/zju/armadillo/armadillo_points.obj");
    // bunny->loadSurface("../../Media/zju/armadillo/armadillo.obj");

    bunny->translate(Vector3f(0.5, 0.2, 0.5));
    bunny->setVisible(true);
    bunny->getElasticitySolver()->setIterationNumber(10);

    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(1, 1, 0));

    //debug code to visualize simulation particles
    // auto m_pointsRender = std::make_shared<PointRenderModule>();
    // m_pointsRender->setColor(Vector3f(0.98, 0.85, 0.40));
    // bunny->addVisualModule(m_pointsRender);

    GLApp window;
    window.createWindow(1024, 768);

    window.mainLoop();

    return 0;
}
