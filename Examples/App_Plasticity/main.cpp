
/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-28
 * @description: Simulate elastoplastic object with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 * @TODO       : simulation is not symmetric, need fixing
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
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
 * setup scene: two cubes fall to the ground, one is elastic, the other is plastic
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);  //scene boundary

    //elastoplastic cube
    std::shared_ptr<ParticleElastoplasticBody<DataType3f>> plastic_cube = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
    root->addParticleSystem(plastic_cube);
    plastic_cube->setMass(1.0);
    plastic_cube->loadParticles(Vector3f(-1.1), Vector3f(1.15), 0.1);
    plastic_cube->loadSurface("../../Media/standard/standard_cube20.obj");
    plastic_cube->scale(0.05);
    plastic_cube->translate(Vector3f(0.3, 0.2, 0.5));
    plastic_cube->setVisible(false);
    plastic_cube->getSurfaceNode()->setVisible(true);
    auto sRender = std::make_shared<SurfaceMeshRender>();
    plastic_cube->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(1, 1, 1));

    //elastic cube
    std::shared_ptr<ParticleElasticBody<DataType3f>> elastic_cube = std::make_shared<ParticleElasticBody<DataType3f>>();
    root->addParticleSystem(elastic_cube);
    elastic_cube->setMass(1.0);
    elastic_cube->loadParticles(Vector3f(-1.1), Vector3f(1.15), 0.1);
    elastic_cube->loadSurface("../../Media/standard/standard_cube20.obj");
    elastic_cube->scale(0.05);
    elastic_cube->translate(Vector3f(0.5, 0.2, 0.5));
    elastic_cube->getElasticitySolver()->setIterationNumber(10);
    elastic_cube->setVisible(false);
    auto sRender2 = std::make_shared<SurfaceMeshRender>();
    elastic_cube->getSurfaceNode()->addVisualModule(sRender2);
    sRender2->setColor(Vector3f(1, 1, 0));
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