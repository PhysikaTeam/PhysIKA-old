/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Simulate fluid mixing
 *               reference <Fast Multiple-fluid Simulation Using Helmholtz Free Energy>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 * @TODO       : Volume conservation of fluid is broken, need fixing
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/MultipleFluidModel.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Rendering/PointRenderModule.h"
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
 * setup scene: multiple fluid blocks fall to the ground and mix
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.02f, true);  //scene boundary

    std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
    root->addParticleSystem(fluid);
    fluid->loadParticles("../../Media/fluid/fluid_point.obj");
    fluid->setMass(100);
    fluid->scale(2);
    fluid->translate(Vector3f(-0.6, -0.3, -0.48));

    //use MultipleFluidModel as the numeric model
    std::shared_ptr<MultipleFluidModel<DataType3f>> multifluid = std::make_shared<MultipleFluidModel<DataType3f>>();
    fluid->setNumericalModel(multifluid);
    fluid->currentPosition()->connect(&multifluid->m_position);
    fluid->currentVelocity()->connect(&multifluid->m_velocity);
    fluid->currentForce()->connect(&multifluid->m_forceDensity);

    auto ptRender1 = std::make_shared<PointRenderModule>();
    ptRender1->setColor(Vector3f(1, 0, 0));
    ptRender1->setColorRange(0, 1);
    fluid->addVisualModule(ptRender1);
    multifluid->m_color.connect(&ptRender1->m_vecIndex);
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
