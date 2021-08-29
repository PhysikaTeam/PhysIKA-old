/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2021-06-17
 * @description: Simulate fluid with PBD using fast algorithm
 *               reference <Position Based Fluids>
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
#include "Dynamics/ParticleSystem/ParticleFluidFast.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Rendering/PointRenderModule.h"
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
 * setup scene: fluid pours into a bowl
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1.5, 1, 1.5));
    scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(-0.5, 0, -0.5), Vector3f(1.5, 2, 1.5), 0.02, true);  //scene boundary
    root->loadSDF("../../Media/bowl/bowl.sdf", false);                           //bowl in scene
    //dummy rigid body to render the bowl
    std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
    root->addRigidBody(rigidbody);
    rigidbody->loadShape("../../Media/bowl/bowl.obj");
    rigidbody->setActive(false);
    rigidbody->setVisible(true);
    auto sRender2 = std::make_shared<SurfaceMeshRender>();
    sRender2->setColor(Vector3f(0.8f, 0.8f, 0.8f));
    rigidbody->getSurface()->addVisualModule(sRender2);
    //fluid
    std::shared_ptr<ParticleFluidFast<DataType3f>> fluid = std::make_shared<ParticleFluidFast<DataType3f>>();
    root->addParticleSystem(fluid);
    fluid->loadParticles(Vector3f(0.3, 0.2, 0.4), Vector3f(0.7, 1.5, 0.6475), 0.005);
    fluid->setMass(100);
    auto ptRender = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(1, 0, 0));
    ptRender->setColorRange(0, 4);
    fluid->addVisualModule(ptRender);
    fluid->currentVelocity()->connect(&ptRender->m_vecIndex);
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
