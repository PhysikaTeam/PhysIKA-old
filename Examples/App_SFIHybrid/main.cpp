/**
 * @author     : n-jing (siliuhe@sina.com)
 * @date       : 2020-07-23
 * @description: Simulate coupling between PBD fluid and elastic bodies with different algorithm backends
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-17
 * @description: poslish code
 * @version    : 1.1
 */

#include <memory>
#include <boost/property_tree/json_parser.hpp>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Framework/Mapping/PointSetToPointSet.h"
#include "Dynamics/ParticleSystem/PositionBasedFluidModel.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/SolidFluidInteraction.h"
#include "Dynamics/EmbeddedMethod/EmbeddedFiniteElement.h"
#include "Dynamics/EmbeddedMethod/EmbeddedMassSpring.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

/**
 * setup scene: dambreak fluid simulated with PBD, 3 elastic bodies simulated with Embedded FEM, embedded Mass-spring,
 * and projective-peridynamics respectively
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.015f, true);  //scene boundary

    //PBD fluid
    std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>();
    root->addParticleSystem(fluid);
    fluid->loadParticles(Vector3f(0), Vector3f(0.5, 1.0, 1.0), 0.015f);
    fluid->setMass(10);
    std::shared_ptr<PositionBasedFluidModel<DataType3f>> pbd = std::make_shared<PositionBasedFluidModel<DataType3f>>();
    pbd->setSmoothingLength(0.02);
    fluid->setNumericalModel(pbd);
    fluid->currentPosition()->connect(&pbd->m_position);
    fluid->currentVelocity()->connect(&pbd->m_velocity);
    fluid->currentForce()->connect(&pbd->m_forceDensity);
    auto ptRender = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(0, 0, 1));
    ptRender->setColorRange(0, 1);
    fluid->addVisualModule(ptRender);

    //sfi
    std::shared_ptr<SolidFluidInteraction<DataType3f>> sfi = std::make_shared<SolidFluidInteraction<DataType3f>>();
    sfi->setInteractionDistance(0.02);
    root->addChild(sfi);
    sfi->addParticleSystem(fluid);

    //elastic bodies
    for (int i = 0; i < 3; i++)
    {
        if (i % 3 == 0)
        {
            //elastic bunny simulated with embedded FEM
            std::shared_ptr<EmbeddedFiniteElement<DataType3f>> bunny = std::make_shared<EmbeddedFiniteElement<DataType3f>>();
            root->addParticleSystem(bunny);
            bunny->setMass(1.0);
            bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
            bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
            bunny->translate(Vector3f(0.75, 0.2, 0.4 + i * 0.3));
            bunny->setVisible(false);
            bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
            bunny->getTopologyMapping()->setSearchingRadius(0.05);
            bunny->getElasticitySolver()->setIterationNumber(10);
            auto sRender = std::make_shared<SurfaceMeshRender>();
            bunny->getSurfaceNode()->addVisualModule(sRender);
            sRender->setColor(Vector3f(1, 1, 0));
            boost::property_tree::ptree pt;
            const std::string           jsonfile_path = "../../Media/bunny/embedded_finite_element_sparse.json";
            read_json(jsonfile_path, pt);
            bunny->init_problem_and_solver(pt);
            sfi->addParticleSystem(bunny);
        }
        else if (i % 3 == 1)
        {
            //elastic bunny simulated with projective-peridynamics
            std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
            root->addParticleSystem(bunny);
            bunny->setMass(1.0);
            bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
            bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
            bunny->translate(Vector3f(0.75, 0.2, 0.4 + i * 0.3));
            bunny->setVisible(false);
            bunny->getElasticitySolver()->setIterationNumber(10);
            bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
            bunny->getTopologyMapping()->setSearchingRadius(0.05);
            auto sRender = std::make_shared<SurfaceMeshRender>();
            bunny->getSurfaceNode()->addVisualModule(sRender);
            sRender->setColor(Vector3f(1, 0, 1));
            sfi->addParticleSystem(bunny);
        }
        else
        {
            //elastic bunny simulated with embedded mass spring method
            std::shared_ptr<EmbeddedMassSpring<DataType3f>> bunny = std::make_shared<EmbeddedMassSpring<DataType3f>>();
            root->addParticleSystem(bunny);
            bunny->setMass(1.0);
            bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
            bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
            bunny->translate(Vector3f(0.75, 0.2, 0.4 + i * 0.3));
            bunny->setVisible(false);
            bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
            bunny->getTopologyMapping()->setSearchingRadius(0.05);
            auto sRender = std::make_shared<SurfaceMeshRender>();
            bunny->getSurfaceNode()->addVisualModule(sRender);
            sRender->setColor(Vector3f(0, 1, 1));
            boost::property_tree::ptree pt;
            const std::string           jsonfile_path = "../../Media/bunny/embedded_mass_spring_sparse.json";
            read_json(jsonfile_path, pt);
            bunny->init_problem_and_solver(pt);
            sfi->addParticleSystem(bunny);
        }
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
