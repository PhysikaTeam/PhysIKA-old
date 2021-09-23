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
#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/PointSet.h"
#include "Framework/Framework/Log.h"

#include "Rendering/PointRenderModule.h"

#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"
#include "Dynamics/ParticleSystem/Peridynamics.h"

#include "Framework/Collision/CollidableSDF.h"
#include "Framework/Collision/CollidablePoints.h"
#include "Framework/Collision/CollisionSDF.h"
#include "Framework/Framework/Gravity.h"
#include "Dynamics/ParticleSystem/FixedPoints.h"
#include "Framework/Collision/CollisionPoints.h"
#include "Dynamics/ParticleSystem/ParticleSystem.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/SolidFluidInteraction.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Rendering/SurfaceMeshRender.h"

#include "Dynamics/EmbeddedMethod/EmbeddedFiniteElement.h"
#include "Dynamics/EmbeddedMethod/EmbeddedMassSpring.h"
#include <boost/property_tree/json_parser.hpp>

using namespace std;
using namespace PhysIKA;

/**
 * @brief setup some basic information for the bunny model.
 *        such as color and elasticity solver.
 * 
 * @tparam T 
 * @param bunny the input model.
 * @param i the index for the current model.
 * @param color the color for the current model.
 * @param offset_dis the displacement offset of the current model.
 */
template <typename T>
void SetupModel(T& bunny, int i, std::string model = "")
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
    bunny->translate(Vector3f(0.2 + 0.32 * i, 0.2, 0.8));
    bunny->setVisible(true);
    bunny->getElasticitySolver()->setIterationNumber(10);
    //bunny->getElasticitySolver()->setHorizon(0.03);
    bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
    bunny->getTopologyMapping()->setSearchingRadius(0.05);
}

/**
 * @brief add simulation model.
 * 
 * @param root the scene instance.
 * @param sfi the collision detection instance.
 * @param i the id of the current added model.
 * @param phy_model the physics model file for the current added model.
 * @param geo_model thee geometry model for the current added model.
 * @param offset_dis the offset for the current added model.
 */
void AddSimulationModel(std::shared_ptr<StaticBoundary<DataType3f>>& root, std::shared_ptr<SolidFluidInteraction<DataType3f>>& sfi, int i, std::string model = "")
{
    if (model == "mass_spring")
    {
        std::shared_ptr<EmbeddedMassSpring<DataType3f>> bunny = std::make_shared<EmbeddedMassSpring<DataType3f>>();
        root->addParticleSystem(bunny);
        SetupModel(bunny, i, model);

        boost::property_tree::ptree pt;
        read_json("../../Media/bunny/collision_hybrid.json", pt);
        bunny->init_problem_and_solver(pt);
        sfi->addParticleSystem(bunny);
    }
    else if (model == "fem")
    {
        std::shared_ptr<EmbeddedFiniteElement<DataType3f>> bunny = std::make_shared<EmbeddedFiniteElement<DataType3f>>();
        root->addParticleSystem(bunny);
        SetupModel(bunny, i, model);

        boost::property_tree::ptree pt;
        if (i == 1)
            read_json("../../Media/bunny/calibration.json", pt);
        else
            read_json("../../Media/bunny/origin.json", pt);

        pt.put("fast_FEM", false);
        bunny->init_problem_and_solver(pt);
        sfi->addParticleSystem(bunny);
    }
    else
    {
        std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();

        root->addParticleSystem(bunny);
        SetupModel(bunny, i, model);
        sfi->addParticleSystem(bunny);
    }
}

/**
 * @brief Create a Scene object. setup scene: 3 elastic bunnies spacing horizontally fall to the ground, using different algorithm backend
 * 
 */
void CreateScene()
{

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1, 2.0, 1));
    scene.setLowerBound(Vector3f(0, 0.0, 0));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0, 0.0, 0), Vector3f(1, 2.0, 1), 0.015f, true);
    //root->loadSDF("box.sdf", true);

    std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
    //

    root->addChild(sfi);
    sfi->setInteractionDistance(0.03);  // 0.02 is an very important parameter

    for (int i = 0; i < 3; i++)
    {

        string model = (i % 3 == 0) ? "mass_spring" : (i % 3 == 1) ? "fem"
                                                                   : "fem";
        AddSimulationModel(root, sfi, i, model);
    }
}

int main()
{
    CreateScene();

    Log::setOutput("console_log.txt");
    Log::setLevel(Log::Info);
    Log::sendMessage(Log::Info, "Simulation begin");

    GLApp window;
    window.createWindow(1024, 768);

    window.mainLoop();

    Log::sendMessage(Log::Info, "Simulation end!");
    return 0;
}