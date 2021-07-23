
/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Simulate dry sand with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 * @TODO       : Simulation result is not correct, find out why.
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/ParticleSystem/GranularModule.h"
#include "Rendering/PointRenderModule.h"
#include "Rendering/RigidMeshRender.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

/**
 * redirect log message to standard output
 */
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
 * A bunny-shaped granular body fall onto a bar
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));
    root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);

    //dummy rigid bar, for rendering
    std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
    root->addRigidBody(rigidbody);
    rigidbody->loadShape("../../Media/bar/bar.obj");
    rigidbody->setActive(false);
    rigidbody->translate(Vector3f(0.2f, 0.2f, 0));  //translation matches sdf
    auto renderModule = std::make_shared<RigidMeshRender>(rigidbody->getTransformationFrame());
    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
    rigidbody->getSurface()->addVisualModule(renderModule);

    //bunny
    std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
    root->addParticleSystem(child);
    child->setMass(1.0);
    child->loadParticles("../../Media/bunny/bunny_points.obj");
    child->loadSurface("../../Media/bunny/bunny_mesh.obj");
    child->translate(Vector3f(0.3, 0.4, 0.5));
    child->setDt(0.001);
    auto elasto = std::make_shared<GranularModule<DataType3f>>();
    elasto->enableFullyReconstruction();
    elasto->setCohesion(0);
    child->setElastoplasticitySolver(elasto);

    auto m_pointsRender = std::make_shared<PointRenderModule>();
    m_pointsRender->setColor(Vector3f(0.98, 0.85, 0.40));
    child->addVisualModule(m_pointsRender);
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
