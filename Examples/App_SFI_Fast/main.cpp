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

#include "Dynamics/ParticleSystem/ParticleFluidFast.h"

#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"

#include "SFIFast.h"

using namespace std;
using namespace PhysIKA;

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
//	scene.setUpperBound(Vector3f(1, 1.0, 0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();

	root->loadCube(Vector3f(0), Vector3f(1), 0.015f, true);
	std::shared_ptr<ParticleFluidFast<DataType3f>> fluid = std::make_shared<ParticleFluidFast<DataType3f>>();
	root->addParticleSystem(fluid);

	fluid->loadParticles(Vector3f(0), Vector3f(0.5, 0.5, 0.5), 0.005f);
	fluid->setMass(10);


	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(1, 0, 0));
	ptRender->setColorRange(0, 4);
	fluid->addVisualModule(ptRender);

	fluid->currentVelocity()->connect(&ptRender->m_vecIndex);

	fluid->setActive(true);

	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(fluid->getTopologyModule());
		auto& points = pSet->getPoints();
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

	std::shared_ptr<ParticleElasticBody<DataType3f>> bunny = std::make_shared<ParticleElasticBody<DataType3f>>();
	//root->addParticleSystem(bunny);
	bunny->setMass(1.0);
	bunny->loadParticles("../../Media/bunny/sparse_bunny_points.obj");
	bunny->loadSurface("../../Media/bunny/sparse_bunny_mesh.obj");
	bunny->scale(1.0f);
	bunny->translate(Vector3f(0.75, 0.2, 0.4 +  0.3));
	bunny->setVisible(false);
	bunny->getElasticitySolver()->setIterationNumber(10);
	bunny->getElasticitySolver()->inHorizon()->setValue(0.03);
	bunny->getTopologyMapping()->setSearchingRadius(0.05);

	auto sRender = std::make_shared<SurfaceMeshRender>();
	bunny->getSurfaceNode()->addVisualModule(sRender);
	sRender->setColor(Vector3f(0.0, 1.0, 1.0));

	root->addParticleSystem(bunny);


	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(bunny->getTopologyModule());
		auto& points = pSet->getPoints();
		HostArray<Vector3f> hpoints(points.size());
		Function1Pt::copy(hpoints, points);

		std::ofstream outf("Particles.obj", ios::app);
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

	std::shared_ptr<SFIFast<DataType3f>> sfi = std::make_shared<SFIFast<DataType3f>>();
	// 
	sfi->addParticleSystem(fluid);
	sfi->addParticleSystem(bunny);

	root->addChild(sfi);
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


