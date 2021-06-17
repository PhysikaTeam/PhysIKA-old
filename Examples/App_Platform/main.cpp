#include <iostream>
#include "Framework/Framework/SceneGraph.h"
//#include "Dynamics/ParticleSystem/ParticleElasticBodyNorm.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"

#include "Dynamics/ParticleSystem/ElasticityModule.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/SemiAnalyticalSFINode.h"
#include "Dynamics/ParticleSystem/SemiAnalyticalIncompressibleFluidModel.h"
#include "Dynamics/ParticleSystem/TriangularSurfaceMeshNode.h"
#include "Dynamics/ParticleSystem/ParticleElasticBody.h"
#include "Dynamics/ParticleSystem/MeshCollision.h"
#include "Dynamics/ParticleSystem/ParticleWriter.h"

#include "Framework/Topology/TriangleSet.h"

#include "GUI/GlutGUI/GLApp.h"

#include <utility>

using namespace PhysIKA;

void creatscene_ca()
{
	//freopen("backup.txt", "w", stdout);
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setTotalTime(3.0f);
	scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));
	scene.setLowerBound(Vector3f(-1.0f, 0.0f, 0.0f));
	scene.setUpperBound(Vector3f(1.0f, 1.0f, 1.0f));
	scene.setFrameRate(100);
	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(0.01), Vector3f(0.99), 0.01f, true);

	std::shared_ptr<SemiAnalyticalSFINode<DataType3f>> sfi = std::make_shared<SemiAnalyticalSFINode<DataType3f>>();
	sfi->setInteractionDistance(0.016);
	

	//Particle fluid node
	std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>("fluid");
	auto mf_pointsRender = std::make_shared<PointRenderModule>();
	fluid->addVisualModule(mf_pointsRender);
	fluid->loadParticles(Vector3f(-0.985, 0.015, 0.015), Vector3f(-0.585f, 0.6f, 0.985), 0.005); // sphere


	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(fluid->getTopologyModule());
		auto& points = pSet->getPoints();
		HostArray<Vector3f> hpoints(points.size());
		Function1Pt::copy(hpoints, points);

		std::ofstream outf("Particles.txt");
		if (outf.is_open())
		{
			for (int i = 0; i < hpoints.size(); ++i)
			{
				Vector3f curp = hpoints[i];
				outf << curp[0] << " " << curp[1] << " " << curp[2] << std::endl;
			}
			outf.close();

			std::cout << " Particle output:  FINISHED." << std::endl;
		}
	}
	
	//fluid->loadParticles(Vector3f(0.25, 0.85, 0.25), Vector3f(0.75f, 0.9f, 0.75), 0.005); // sphere


 	mf_pointsRender->setColor(Vector3f(0, 1, 1));
 	mf_pointsRender->setColorRange(0, 3);
	sfi->getParticleVelocity()->connect(&mf_pointsRender->m_vecIndex);


	//Boundary 1
	auto boundary1 = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("boundary1");
	boundary1->getTriangleSet()->loadObjFile("../../Media/platform/spp2.obj");
	boundary1->scale(1/1000.0);
	boundary1->translate(Vector3f(0.5, 0.0,0.5));


	auto sRenderf = std::make_shared<SurfaceMeshRender>();
	boundary1->addVisualModule(sRenderf);
	sRenderf->setColor(Vector3f(1, 1, 0));
	sRenderf->setVisible(true);
	//sRenderf->setDisplayMode(DisplayMode::WireFrameMode);

	//Boundary 2;
	auto boundary2 = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("boundary2");
	boundary2->getTriangleSet()->loadObjFile("../../Media/standard/standard_cube2.obj");
	//boundary2->getTriangleSet()->loadObjFile("../../Media/test_mesh/test_128.obj");
	//boundary2->scale(2.0);

	auto sRenderf2 = std::make_shared<SurfaceMeshRender>();
	boundary2->addVisualModule(sRenderf2);
	sRenderf2->setColor(Vector3f(1, 1, 0));
	sRenderf2->setVisible(false);
	//sRenderf2->setDisplayMode(DisplayMode::WireFrameMode);
	

	//Elastic object
	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	//root->addParticleSystem(bunny);

	

	auto sRender = std::make_shared<SurfaceMeshRender>();
	
	bunny->getElasticitySolver()->setIterationNumber(10);

	sfi->addParticleSystem(fluid);
	//sfi->addTriangularSurfaceMesh(bunny->getSurfaceNode());
	sfi->addTriangularSurfaceMeshNode(boundary1);
	sfi->addTriangularSurfaceMeshNode(boundary2);
//	sfi->addParticleSystem(bunny);
	
	//root->addParticleSystem(fluid);
	root->addChild(sfi);

	
	

	GLApp window;
	window.createWindow(1024, 768);
	window.mainLoop();
	
}

int main()
{
	creatscene_ca();
	return 0;
}

