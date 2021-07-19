/**
 * @author     : Chen Xiaosong (xiaosong0911@gmail.com)
 * @date       : 2021-03-25
 * @description: demo of multi-phase SPH solver
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-19
 * @description: poslish code
 * @version    : 1.1
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/FastMultiphaseSPH/FastMultiphaseSPH.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/MultipleFluidModel.h"
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
 * setup scene: a block falls into fluid
 *
 * @param[in] dissolution  whether the block can dissolute
 */
void createSceneBlock(int dissolution)
{
    SceneGraph& scene = SceneGraph::getInstance();

    auto root        = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();
    using particle_t = FastMultiphaseSPH<DataType3f>::particle_t;
    root->loadParticlesAABBSurface(Vector3f(-1.02, -0.02, -1.02), Vector3f(1.02, 2.5, 1.02), root->getSpacing(), particle_t::BOUDARY);
    root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -0.5), Vector3f(0, 0.8, 0.5), root->getSpacing(), particle_t::FLUID);
    root->loadParticlesAABBVolume(Vector3f(0.2, 0., -0.2), Vector3f(0.8, 0.6, 0.2), root->getSpacing(), particle_t::SAND);
    root->setDissolutionFlag(dissolution);
    root->initSync();

    auto ptRender1 = std::make_shared<PointRenderModule>();
    ptRender1->setColor(Vector3f(1, 0, 1));
    ptRender1->setColorRange(0, 1);
    root->addVisualModule(ptRender1);

    root->m_phase_concentration.connect(&ptRender1->m_vecIndex);
}

/**
 * setup scene: a toy-shaped object falls into fluid
 *
 * @param[in] dissolution  whether the object can dissolute
 */
void createSceneToy(int dissolution)
{
    SceneGraph& scene = SceneGraph::getInstance();

    auto root        = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();
    using particle_t = FastMultiphaseSPH<DataType3f>::particle_t;
    root->loadParticlesAABBSurface(Vector3f(-1.02, -0.02, -1.02), Vector3f(1.02, 2.5, 1.02), root->getSpacing(), particle_t::BOUDARY);
    root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -1.0), Vector3f(1.0, 0.2, 1.0), root->getSpacing(), particle_t::FLUID);
    root->loadParticlesFromFile("../../Media/toy.obj", particle_t::SAND);
    root->setDissolutionFlag(dissolution);
    root->initSync();

    auto ptRender1 = std::make_shared<PointRenderModule>();
    ptRender1->setColor(Vector3f(1, 0, 1));
    ptRender1->setColorRange(0, 1);
    root->addVisualModule(ptRender1);

    root->m_phase_concentration.connect(&ptRender1->m_vecIndex);
}

/**
 * setup scene: a crag-shaped object falls into fluid
 *
 * @param[in] dissolution  whether the object can dissolute
 */
void createSceneCrag(int dissolution)
{
    SceneGraph& scene = SceneGraph::getInstance();

    auto root        = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();
    using particle_t = FastMultiphaseSPH<DataType3f>::particle_t;
    root->loadParticlesAABBSurface(Vector3f(-1.02, -0.02, -1.02), Vector3f(1.02, 2.5, 1.02), root->getSpacing(), particle_t::BOUDARY);
    root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -1.0), Vector3f(1.0, 0.2, 1.0), root->getSpacing(), particle_t::FLUID);
    root->loadParticlesFromFile("../../Media/crag.obj", particle_t::SAND);
    root->setDissolutionFlag(dissolution);
    root->initSync();

    auto ptRender1 = std::make_shared<PointRenderModule>();
    ptRender1->setColor(Vector3f(1, 0, 1));
    ptRender1->setColorRange(0, 1);
    root->addVisualModule(ptRender1);

    root->m_phase_concentration.connect(&ptRender1->m_vecIndex);
}

/**
 * setup different scenes according to command line options
 * usage:
 *  excutable-name scene-name dissolution-flag
 * sample:
 *  App_MultiphaseSPH.exe toy 1
 *
 * scene-name options: block, toy, crag
 * dissolution-flag options: 0, 1
 */
int main(int argc, char** argv)
{
    std::string scene       = "block";
    int         dissolution = 1;
    if (argc >= 2)
        scene = argv[1];
    if (argc >= 3)
        dissolution = atoi(argv[2]);

    if (scene == "block")
        createSceneBlock(dissolution);
    else if (scene == "toy")
        createSceneToy(dissolution);
    else if (scene == "crag")
        createSceneCrag(dissolution);
    else
    {
        printf("unknown scene name");
        exit(1);
    }

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
