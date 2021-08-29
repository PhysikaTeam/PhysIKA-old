/**
 * @author     : n-jing (siliuhe@sina.com)
 * @date       : 2020-06-30
 * @description: demo of embedded mass-spring and embedded fem methods
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-19
 * @description: poslish code
 * @version    : 1.1
 */
#include <memory>

#include <boost/property_tree/json_parser.hpp>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/SolidFluidInteraction.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Dynamics/EmbeddedMethod/EmbeddedFiniteElement.h"
#include "Dynamics/EmbeddedMethod/EmbeddedMassSpring.h"
#include "Rendering/SurfaceMeshRender.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

/**
 * setup properties for the object to be simulated
 *
 * @param[in] bunny  the scene node representing the elastic object
 * @param[in] i      index of the elastic object
 */
template <typename T>
void setupModel(T& bunny, int i)
{
    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);

    if (i == 0)
        sRender->setColor(Vector3f(1, 1, 0));
    else if (i == 1)
        sRender->setColor(Vector3f(1, 0, 1));
    else if (i == 2)
        sRender->setColor(Vector3f(0, 1, 1));
    else if (i == 3)
        sRender->setColor(Vector3f(0, 0, 1));
    else if (i == 4)
        sRender->setColor(Vector3f(0, 1, 0));
    else
        sRender->setColor(Vector3f(1, 0, 0));

    bunny->setMass(1.0);
    bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
    bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
    bunny->translate(Vector3f(0.4, 0.2 + i * 0.2, 0.8));
    bunny->setVisible(true);
    bunny->getElasticitySolver()->setIterationNumber(10);
    bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
    bunny->getTopologyMapping()->setSearchingRadius(0.05);
}

/**
 * add elastic body in scene
 *
 * @param[in] root  root node of current scene graph
 * @param[in] sfi   sfi node in scene, elastic bodies are registered to the node for collision handling
 * @param[in] i     index of the elastic body, setup model color and position according to this index
 * @param[in] model name of the simulation method backend
 */
void addSimulationModel(std::shared_ptr<StaticBoundary<DataType3f>>& root, std::shared_ptr<SolidFluidInteraction<DataType3f>>& sfi, int i, std::string model = "")
{
    if (model == "mass_spring")
    {
        std::shared_ptr<EmbeddedMassSpring<DataType3f>> bunny = std::make_shared<EmbeddedMassSpring<DataType3f>>();
        root->addParticleSystem(bunny);
        setupModel(bunny, i);
        boost::property_tree::ptree pt;
        read_json("../../Media/bunny/collision_hybrid.json", pt);
        bunny->init_problem_and_solver(pt);
        sfi->addParticleSystem(bunny);
    }
    else if (model == "fem")
    {
        std::shared_ptr<EmbeddedFiniteElement<DataType3f>> bunny = std::make_shared<EmbeddedFiniteElement<DataType3f>>();
        root->addParticleSystem(bunny);
        setupModel(bunny, i);
        boost::property_tree::ptree pt;
        read_json("../../Media/bunny/collision_hybrid.json", pt);
        bunny->init_problem_and_solver(pt);
        sfi->addParticleSystem(bunny);
    }
    else
    {
        std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
        root->addParticleSystem(bunny);
        setupModel(bunny, i);
        sfi->addParticleSystem(bunny);
    }
}

/**
 * setup scene: 6 elastic bunnies spacing vertically fall to the ground, using different algorithm backend
 */
void createScene()
{

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(7.0, 4.0, 3.0));
    scene.setLowerBound(Vector3f(-3.0, 0.0, -1.0));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(-3.0, 0.0, -1.0), Vector3f(7.0, 4.0, 3.0), 0.015f, true);  //scene boundary

    //use SFI node to handle particle collisions between elastic bodies
    std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
    root->addChild(sfi);
    sfi->setInteractionDistance(0.03);  // 0.03 is an very important parameter

    //6 bunnies: mass-spring, fem, projective-peridynamics, and repeat
    for (int i = 0; i < 6; i++)
    {
        string model = (i % 3 == 0) ? "mass_spring" : ((i % 3 == 1) ? "fem" : "");
        addSimulationModel(root, sfi, i, model);
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
