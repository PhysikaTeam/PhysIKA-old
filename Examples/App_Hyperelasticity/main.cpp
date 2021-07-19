
/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-20
 * @description: Comparison of hyperelasticity and elasticity with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
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
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/HyperelasticityModule.h"
#include "Rendering/SurfaceMeshRender.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

void RecieveLogMessage(const Log::Message& m)
{
    switch (m.type)
    {
        case Log::Info:
            cout << ">>>: " << m.text << endl;
            break;
        case Log::Warning:
            cout << "???: " << m.text << endl;
            break;
        case Log::Error:
            cout << "!!!: " << m.text << endl;
            break;
        case Log::User:
            cout << ">>>: " << m.text << endl;
            break;
        default:
            break;
    }
}

/**
 * setup scene: two bunnies fall to the ground, one with hyperelastic constitutive model, the other
 * with default elastic constitutive model
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);  //scene boundary

    //hyperelastic bunny
    std::shared_ptr<ParticleElasticBody<DataType3f>> hyper_bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
    root->addParticleSystem(hyper_bunny);
    hyper_bunny->setMass(1.0);
    hyper_bunny->loadParticles("../../Media/bunny/bunny_points.obj");
    hyper_bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
    hyper_bunny->translate(Vector3f(0.3, 0.2, 0.5));
    hyper_bunny->setVisible(false);
    auto hyper_model = std::make_shared<HyperelasticityModule<DataType3f>>();
    hyper_model->setEnergyFunction(HyperelasticityModule<DataType3f>::Quadratic);
    hyper_bunny->setElasticitySolver(hyper_model);
    hyper_bunny->getElasticitySolver()->setIterationNumber(10);
    auto sRender = std::make_shared<SurfaceMeshRender>();
    hyper_bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(1, 1, 0));

    //elastic bunny
    std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
    root->addParticleSystem(bunny);
    bunny->setMass(1.0);
    bunny->loadParticles("../../Media/bunny/bunny_points.obj");
    bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
    bunny->translate(Vector3f(0.7, 0.2, 0.5));
    bunny->setVisible(false);
    auto sRender2 = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender2);
}

int main()
{
    createScene();

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
