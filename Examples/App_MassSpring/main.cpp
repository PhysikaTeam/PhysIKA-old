#include <iostream>
#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/EmbeddedMethod/EmbeddedMassSpring.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include <boost/property_tree/json_parser.hpp>

#include "Framework/Topology/TriangleSet.h"

using namespace PhysIKA;
using namespace std;

int main()
{
    Log::sendMessage(Log::Info, "Simulation start");

    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);
    std::shared_ptr<EmbeddedMassSpring<DataType3f>> bunny = std::make_shared<EmbeddedMassSpring<DataType3f>>();

    root->addParticleSystem(bunny);

    //auto m_pointsRender = std::make_shared<PointRenderModule>();
    //m_pointsRender->setColor(Vector3f(0, 1, 1));
    //bunny->addVisualModule(m_pointsRender);

    bunny->setMass(1.0);
    /*bunny->loadParticles("../../Media/bunny/bunny_points.obj");
    bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");*/

    const string particles_file = "../../Media/dragon/dragon_points_10570.obj";
    bunny->loadParticles(particles_file);
    bunny->loadSurface("../../Media/dragon/dragon.obj");

    // bunny->scale(1.0 / 6);
    bunny->translate(Vector3f(0.5, 0.2, 0.5));
    bunny->setVisible(true);

    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(1, 1, 0));

    // Output all particles to .txt file.
    {
        auto                pSet   = TypeInfo::CastPointerDown<PointSet<DataType3f>>(bunny->getTopologyModule());
        auto&               points = pSet->getPoints();
        HostArray<Vector3f> hpoints(points.size());
        Function1Pt::copy(hpoints, points);

        std::ofstream outf("Particles.obj");
        if (outf.is_open())
        {
            for (int i = 0; i < hpoints.size(); ++i)
            {
                Vector3f curp = hpoints[i];
                outf << "v " << curp[0] << " " << curp[1] << " " << curp[2] << std::endl;
            }
            outf.close();

            std::cout << " Particle output:  FINISHED." << std::endl;
        }
    }

    // bunny->getElasticitySolver()->setIterationNumber(10);
    boost::property_tree::ptree pt;
    const std::string           jsonfile_path = "../../Media/dragon/embedded_mass_spring.json";
    read_json(jsonfile_path, pt);
    bunny->init_problem_and_solver(pt);

    GLApp window;
    window.createWindow(1024, 768);

    window.mainLoop();

    return 0;
}
