
/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Simulate fracture with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 * @TODO       : fix the issue of particle explosion
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/FractureModule.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Rendering/PointRenderModule.h"
#include "Rendering/RigidMeshRender.h"
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
 * setup simulation scene: a cuboid object fractures due to bending over a bar
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setLowerBound(Vector3f(0, 0, 0));
    scene.setUpperBound(Vector3f(1, 0.5, 0.5));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);  //static bar obstacle
    root->translate(Vector3f(0.1f, 0.2f, 0));
    //dummy rigid bar, for rendering
    std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
    root->addRigidBody(rigidbody);
    rigidbody->loadShape("../../Media/bar/bar.obj");
    rigidbody->setActive(false);
    rigidbody->translate(Vector3f(0.1f, 0.2f, 0));
    auto renderModule = std::make_shared<RigidMeshRender>(rigidbody->getTransformationFrame());
    renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
    rigidbody->getSurface()->addVisualModule(renderModule);
    //scene boundary
    root->loadCube(Vector3f(0), Vector3f(1, 0.5, 0.5), 0.005, true);

    //elastoplastic cuboid that could fracture
    std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
    root->addParticleSystem(child);
    child->setMass(1.0);
    child->loadParticles(Vector3f(0, 0.25, 0.1), Vector3f(0.3f, 0.4, 0.4), 0.005f);
    auto fracture = std::make_shared<FractureModule<DataType3f>>();
    child->setElastoplasticitySolver(fracture);
    auto m_pointsRender = std::make_shared<PointRenderModule>();
    m_pointsRender->setColor(Vector3f(0, 1, 1));
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
