/**
 * @author     : n-jing (siliuhe@sina.com)
 * @date       : 2020-06-30
 * @description: demo of Material Consistant
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
#include "Dynamics/FiniteElementMethod/Common/FEMCommonKConsistent.h"

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
 * @brief init problem and solver.
 * 
 * @tparam DynamicsT the dynamics backend.
 * @param model the current model.
 * @param pt the property tree for config the problem and solver.
 */
template <typename DynamicsT>
void init_problem_and_solver(const std::shared_ptr<DynamicsT>& model, const boost::property_tree::ptree& pt)
{
    model->init_problem_and_solver(pt);
}

template <>
void init_problem_and_solver(const std::shared_ptr<ParticleElasticBody<DataType3f>>& model, const boost::property_tree::ptree& pt)
{
}

/**
 * @brief add simulation model.
 * 
 * @tparam DynamicsT the current model.
 * @param root the scene instance.
 * @param sfi the collision detection instance.
 * @param color the color for the current model.
 * @param pos the position for the current model.
 * @param pt the property tree for config the problem and solver.
 * @return std::shared_ptr<DynamicsT> 
 */
template <typename DynamicsT>
std::shared_ptr<DynamicsT> AddSimulationModel(std::shared_ptr<StaticBoundary<DataType3f>>& root, std::shared_ptr<SolidFluidInteraction<DataType3f>>& sfi, const Vector3f& color, const Vector3f& pos, const boost::property_tree::ptree& pt)
{
    auto model = std::make_shared<DynamicsT>();
    root->addParticleSystem(model);

    auto sRender = std::make_shared<SurfaceMeshRender>();
    sRender->setColor(color);
    model->getSurfaceNode()->addVisualModule(sRender);

    model->setMass(1);
    model->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
    model->loadSurface("../../Media/bunny/bunny_mesh.obj");
    model->translate(pos);
    model->setVisible(true);
    model->getElasticitySolver()->setIterationNumber(10);
    model->getElasticitySolver()->inHorizon()->setValue(0.03);
    model->getTopologyMapping()->setSearchingRadius(0.05);
    sfi->addParticleSystem(model);
    init_problem_and_solver<DynamicsT>(model, pt);
    return model;
}

template <typename DynamicsT>
std::shared_ptr<DynamicsT> AddSimulationModel(std::shared_ptr<StaticBoundary<DataType3f>>& root, std::shared_ptr<SolidFluidInteraction<DataType3f>>& sfi, const Vector3f& color, const Vector3f& pos, const std::string& jsonfile)
{
    boost::property_tree::ptree pt;
    read_json(jsonfile, pt);
    return AddSimulationModel<DynamicsT>(root, sfi, color, pos, pt);
}

void CreateScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(2, 2.0, 2));
    scene.setLowerBound(Vector3f(0, 0.0, 0));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0, 0.0, 0), Vector3f(2, 2.0, 2), 0.015f, true);

    std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();

    root->addChild(sfi);
    sfi->setInteractionDistance(0.03);  // 0.02 is an very important parameter

    auto EMS = AddSimulationModel<EmbeddedMassSpring<DataType3f>>(root, sfi, Vector3f(1, 1, 0), Vector3f(0.2 + 0.42 * 0, 0.6, 0.8), "../../Media/zju/consistent/consistent.json");

    auto EFEM = AddSimulationModel<EmbeddedFiniteElement<DataType3f>>(root, sfi, Vector3f(1, 0, 0), Vector3f(0.2 + 0.42 * 2, 0.6, 0.8), "../../Media/zju/consistent/origin.json");

    boost::property_tree::ptree pt;
    read_json("../../Media/zju/consistent/origin.json", pt);
    auto ratio = k_consistent<Real>(EMS->epb_fac_->get_K(), EMS->epb_fac_->get_mass_vec(), EFEM->epb_fac_->get_K(), EFEM->epb_fac_->get_mass_vec(), 6, pt.get<int>("consistent_num", 6));
    std::cout << "ratio: " << ratio << std::endl;

    auto young = pt.get_child("physics").get<double>("Young");
    std::cout << "young: " << young << std::endl;
    pt.get_child("physics").put("Young", young * ratio);
    young = pt.get_child("physics").get<double>("Young");
    std::cout << "young: " << young << std::endl;
    //exit(1);
    auto FEFEM = AddSimulationModel<EmbeddedFiniteElement<DataType3f>>(root, sfi, Vector3f(1, 0, 1), Vector3f(0.2 + 0.42 * 1, 0.6, 0.8), pt);
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