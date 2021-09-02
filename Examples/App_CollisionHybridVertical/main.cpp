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
#include "Dynamics/FiniteElementMethod/Geometry/surf2tet.h"

using namespace std;
using namespace PhysIKA;

template <typename T>
void SetupModel(T& bunny, int i, Vector3f color, const float offset_dis)
{
    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(color);
    //
    //if (model == "mass_spring")
    //  sRender->setColor(Vector3f(1, 1, 0));
    //else if (model == "fem")
    //  sRender->setColor(Vector3f(1, 0, 1));
    //else
    //  sRender->setColor(Vector3f(0, 1, 1));

    bunny->setMass(1000.0);

    bunny->translate(Vector3f(0.3 + i % 2 * 0.4, 0.2 + offset_dis, 0.8));
    bunny->setVisible(true);
    bunny->getElasticitySolver()->setIterationNumber(10);
    //bunny->getElasticitySolver()->setMu(1e20);
    //bunny->getElasticitySolver()->setLambda(1e20);

    //bunny->getElasticitySolver()->setHorizon(0.03);
    bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
    bunny->getTopologyMapping()->setSearchingRadius(0.05);
}

void AddSimulationModel(std::shared_ptr<StaticBoundary<DataType3f>>& root, std::shared_ptr<SolidFluidInteraction<DataType3f>>& sfi, int i, std::string phy_model, std::string geo_model, const float offset_dis)
{
    const string    path        = "../../Media/zju/" + geo_model + "/";
    Eigen::Vector3f color_eigen = Eigen::Vector3f::Random();
    double          HI          = 1;  // set HI and LO according to your problem.
    double          LO          = 0;
    double          range       = HI - LO;
    color_eigen                 = (color_eigen + Eigen::Vector3f::Ones()) * range / 2.;  // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    color_eigen                 = (color_eigen + Eigen::Vector3f::Constant(LO));         //set LO as the lower bound (offset)

    Vector3f color(color_eigen(0), color_eigen(1), color_eigen(2));
    if (phy_model == "mass_spring")
    {
        std::shared_ptr<EmbeddedMassSpring<DataType3f>> bunny = std::make_shared<EmbeddedMassSpring<DataType3f>>();
        root->addParticleSystem(bunny);
        const std::string jsonfile_path  = path + phy_model + ".json";
        const string      particles_file = path + geo_model + "_points.obj";
        bunny->loadParticles(particles_file);
        bunny->loadSurface(path + geo_model + ".obj");

        //Vector3f color(1, 1, 0);
        SetupModel(bunny, i, color, offset_dis);

        boost::property_tree::ptree pt;
        read_json(jsonfile_path, pt);

        bunny->init_problem_and_solver(pt);
        sfi->addParticleSystem(bunny);
    }
    else if (phy_model == "fem_tet" || phy_model == "fem_hex" || phy_model == "fem_vox" || phy_model == "fem_hybrid")
    {
        std::shared_ptr<EmbeddedFiniteElement<DataType3f>> bunny = std::make_shared<EmbeddedFiniteElement<DataType3f>>();
        root->addParticleSystem(bunny);

        const string particles_file = path + geo_model + "_points.obj";
        const string surf_file      = path + geo_model + ".obj";
        bunny->loadParticles(particles_file);
        bunny->loadSurface(surf_file);

        SetupModel(bunny, i, color, offset_dis);

        boost::property_tree::ptree pt;
        /*read_json("../../Media/dragon/collision_hybrid.json", pt);*/

        //const std::string jsonfile_path = "../../Media/dragon/embedded_finite_element.json";
        const std::string jsonfile_path = path + phy_model + ".json";
        read_json(jsonfile_path, pt);

        if (phy_model == "fem_tet" && pt.get<bool>("gen_tet", true))
        {
            string tet_file = pt.get<string>("filename") + "coarse_tmp";
            surf2tet(surf_file, tet_file);
            pt.put("filename_coarse", tet_file);
            cout << "after tetrahedronlize " << pt.get<string>("filename_coarse");
        }

        bunny->init_problem_and_solver(pt);
        sfi->addParticleSystem(bunny);
    }
    else if (phy_model == "meshless")
    {
        std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
        root->addParticleSystem(bunny);

        const string particles_file = path + geo_model + "_points.obj";
        bunny->loadParticles(particles_file);
        bunny->loadSurface(path + geo_model + ".obj");
        //Vector3f color(0, 1, 1);
        SetupModel(bunny, i, color, offset_dis);
        sfi->addParticleSystem(bunny);
    }
}

void CreateScene()
{

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1, 4.0, 1));
    scene.setLowerBound(Vector3f(0, 0.0, 0));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0, 0.0, 0), Vector3f(1, 4.0, 1), 0.015f, true);
    //root->loadSDF("box.sdf", true);

    std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
    //

    root->addChild(sfi);
    sfi->setInteractionDistance(0.03);  // 0.02 is an very important parameter
    //dragon 0.014
    //bunny 0.03

    AddSimulationModel(root, sfi, 0, "fem_hex", "david", 0);
    AddSimulationModel(root, sfi, 1, "mass_spring", "bunny", 0);
    AddSimulationModel(root, sfi, 2, "fem_hybrid", "duck", 0.4);
    AddSimulationModel(root, sfi, 3, "meshless", "woodenfish", 0.4);
    AddSimulationModel(root, sfi, 4, "fem_tet", "homer", 0.8);
    AddSimulationModel(root, sfi, 5, "meshless", "armadillo", 0.8);
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
