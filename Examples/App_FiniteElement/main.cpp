#include <iostream>
#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/SceneGraph.h"
#include "Dynamics/EmbeddedMethod/EmbeddedFiniteElement.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include <boost/property_tree/json_parser.hpp>
#include "Dynamics/FiniteElementMethod/Geometry/FEMGeometrySurf2Tet.h"

#include "Framework/Topology/TriangleSet.h"

using namespace PhysIKA;
using namespace std;

//int main()
//{
//	Log::sendMessage(Log::Info, "Simulation start");
//
//	SceneGraph& scene = SceneGraph::getInstance();
//
//	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
//	root->loadCube(Vector3f(0), Vector3f(1), 0.005f, true);
//	std::shared_ptr<EmbeddedFiniteElement<DataType3f>> bunny = std::make_shared<EmbeddedFiniteElement<DataType3f>>();
//
//	root->addParticleSystem(bunny);
//
//	//auto m_pointsRender = std::make_shared<PointRenderModule>();
//	//m_pointsRender->setColor(Vector3f(0, 1, 1));
//	//bunny->addVisualModule(m_pointsRender);
//
//	bunny->setMass(1.0);
//
//	//bunny->loadParticles("../../Media/bunny/bunny_points.obj");
//	//bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
//
//
//	/*const string particles_file = "../../Media/dragon/dragon_points_1190.obj";
//	bunny->loadParticles(particles_file);
//	bunny->loadSurface("../../Media/dragon/dragon.obj");*/
//
//	const string geo_model = "homer";
//	const string path = "../../Media/zju/" + geo_model + "/";
//
//	const string particles_file = path + geo_model + "_points.obj";
//	const string surf_file = path + geo_model + ".obj";
//	bunny->loadParticles(particles_file);
//	bunny->loadSurface(surf_file);
//
//
//
//	//const string particles_file = "../../Media/zju/armadillo/armadillo_points.obj";
//	//bunny->loadParticles(particles_file);
//	//bunny->loadSurface("../../Media/zju/armadillo/armadillo.obj");
//
//  // bunny->scale(1.0 / 6);
//	bunny->translate(Vector3f(0.5, 0.3, 0.5));
//	bunny->setVisible(true);
//
//	// Output all particles to .txt file.
//	{
//		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(bunny->getTopologyModule());
//		auto& points = pSet->getPoints();
//		HostArray<Vector3f> hpoints(points.size());
//		Function1Pt::copy(hpoints, points);
//
//		std::ofstream outf("Particles.obj");
//		if (outf.is_open())
//		{
//			for (int i = 0; i < hpoints.size(); ++i)
//			{
//				Vector3f curp = hpoints[i];
//				outf << "v " << curp[0] << " " << curp[1] << " " << curp[2] << std::endl;
//			}
//			outf.close();
//
//			std::cout << " Particle output:  FINISHED." << std::endl;
//		}
//	}
//
//	auto sRender = std::make_shared<SurfaceMeshRender>();
//	bunny->getSurfaceNode()->addVisualModule(sRender);
//	sRender->setColor(Vector3f(1, 1, 0));
//
//	// bunny->getElasticitySolver()->setIterationNumber(10);
//  boost::property_tree::ptree pt;
// /* const std::string jsonfile_path = "../../Media/bunny/embedded_finite_element.json";*/
// /* const std::string jsonfile_path = "../../Media/dragon/embedded_finite_element.json";*/
//    /*const std::string jsonfile_path = "../../Media/zju/armadillo/fem_tet.json";*/
//    const string phy_model = "fem_tet";
//	const std::string jsonfile_path = path + phy_model + ".json";
//
//  read_json(jsonfile_path, pt);
//
//if(phy_model == "fem_tet" && pt.get<bool>("gen_tet", true)){
//	string tet_file = pt.get<string>("filename_coarse") + "tmp";
//	surf2tet(surf_file, tet_file);
//	pt.put("filename_coarse", tet_file);
//	cout << "after tetrahedronlize " << pt.get<string>("filename_coarse");
//
//}
//
//  bunny->init_problem_and_solver(pt);
//	GLApp window;
//	window.createWindow(1024, 768);
//
//	window.mainLoop();
//
//	return 0;
//}

template <typename T>
void SetupModel(T& bunny, int i, Vector3f color, const float offset_dis)
{
    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(color);

    bunny->setMass(1000.0);

    bunny->translate(Vector3f(0.5, 0.2 + offset_dis, 0.8));
    bunny->setVisible(true);
    bunny->getElasticitySolver()->setIterationNumber(10);
    //bunny->getElasticitySolver()->setMu(1e20);
    //bunny->getElasticitySolver()->setLambda(1e20);

    bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
    /*bunny->getTopologyMapping()->setSearchingRadius(0.05);*/
}

void AddSimulationModel(std::shared_ptr<StaticBoundary<DataType3f>>& root, int i, std::string phy_model, std::string geo_model, const float offset_dis)
{
    const string    path        = "../../Media/zju/" + geo_model + "/";
    Eigen::Vector3f color_eigen = Eigen::Vector3f::Random();
    double          HI          = 1;  // set HI and LO according to your problem.
    double          LO          = 0;
    double          range       = HI - LO;
    color_eigen                 = (color_eigen + Eigen::Vector3f::Ones()) * range / 2.;  // add 1 to the matrix to have values between 0 and 2; multiply with range/2
    color_eigen                 = (color_eigen + Eigen::Vector3f::Constant(LO));         //set LO as the lower bound (offset)

    Vector3f color(color_eigen(0), color_eigen(1), color_eigen(2));
    if (phy_model == "fem_tet" || phy_model == "fem_hex" || phy_model == "fem_vox" || phy_model == "fem_hybrid")
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

        {
            if (phy_model == "fem_tet" && pt.get<bool>("gen_tet", true))
            {
                string tet_file = pt.get<string>("filename") + "tet_tmp";
                surf2tet(surf_file, tet_file);
                pt.put("filename_coarse", tet_file);
                cout << "after tetrahedronlize " << pt.get<string>("filename_coarse");
            }
        }

        bunny->init_problem_and_solver(pt);

        root->addParticleSystem(bunny);
    }
}

void CreateScene()
{

    SceneGraph& scene = SceneGraph::getInstance();
    scene.setUpperBound(Vector3f(1, 10.0, 1));
    scene.setLowerBound(Vector3f(0, 0.0, 0));

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0, 0.0, 0), Vector3f(1, 10.0, 1), 0.015f, true);
    //root->loadSDF("box.sdf", true);

    AddSimulationModel(root, 0, "fem_tet", "bunny", 0);
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
