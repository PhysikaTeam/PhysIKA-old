#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/FastMultiphaseSPH/FastMultiphaseSPH.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/MultipleFluidModel.h"

#include "Rendering/PointRenderModule.h"

#include "Framework/Topology/PointSet.h"

using namespace std;
using namespace PhysIKA;


void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break;
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();

	auto root = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();

	using particle_t = FastMultiphaseSPH<DataType3f>::particle_t;

	root->loadParticlesAABBSurface(Vector3f(-1.02, -0.02, -1.02), Vector3f(1.02, 2.5, 1.02), root->getSpacing(), particle_t::BOUDARY);
	

	//root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -0.5), Vector3f(0, 0.8, 0.5), root->getSpacing(), particle_t::FLUID);
	//root->loadParticlesAABBVolume(Vector3f(0.2, 0., -0.2), Vector3f(0.8, 0.6, 0.2), root->getSpacing(), particle_t::SAND);

	root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -1.0), Vector3f(1.0, 0.2, 1.0), root->getSpacing(), particle_t::FLUID);
	//root->loadParticlesFromFile("../../../Media/toy.obj", particle_t::SAND);
	root->loadParticlesFromFile("../../../Media/crag.obj", particle_t::SAND);

	root->setDissolutionFlag(1);
	root->initSync();

	//std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	//root->loadCube(Vector3f(0), Vector3f(1), 0.02f, true);

	//std::shared_ptr<ParticleFluid<DataType3f>> child1 = std::make_shared<ParticleFluid<DataType3f>>();
	//root->addParticleSystem(child1);

	auto ptRender1 = std::make_shared<PointRenderModule>();
	ptRender1->setColor(Vector3f(1, 0, 1));
	ptRender1->setColorRange(0, 1);
	root->addVisualModule(ptRender1);

	root->m_phase_concentration.connect(&ptRender1->m_vecIndex);

	//root->loadParticles("../../Media/fluid/fluid_point.obj");
	//root->setMass(100);
	//root->scale(2);
	//root->translate(Vector3f(-0.6, -0.3, -0.48));

	//std::shared_ptr<MultipleFluidModel<DataType3f>> multifluid = std::make_shared<MultipleFluidModel<DataType3f>>();
	//child1->currentPosition()->connect(&multifluid->m_position);
	//child1->currentVelocity()->connect(&multifluid->m_velocity);
	//child1->currentForce()->connect(&multifluid->m_forceDensity);
	//multifluid->m_color.connect(&ptRender1->m_vecIndex);

	//child1->setNumericalModel(multifluid);

	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(root->getTopologyModule());
		auto& points = pSet->getPoints();
		HostArray<Vector3f> hpoints(points.size());
		Function1Pt::copy(hpoints, points);

		ofstream outf("Particles.txt");
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
}

void CreateSceneBlock(int dissolution)
{
	SceneGraph& scene = SceneGraph::getInstance();

	auto root = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();

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

	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(root->getTopologyModule());
		auto& points = pSet->getPoints();
		HostArray<Vector3f> hpoints(points.size());
		Function1Pt::copy(hpoints, points);

		ofstream outf("Particles.txt");
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
}

void CreateSceneToy(int dissolution)
{
	SceneGraph& scene = SceneGraph::getInstance();

	auto root = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();

	using particle_t = FastMultiphaseSPH<DataType3f>::particle_t;

	root->loadParticlesAABBSurface(Vector3f(-1.02, -0.02, -1.02), Vector3f(1.02, 2.5, 1.02), root->getSpacing(), particle_t::BOUDARY);

	root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -1.0), Vector3f(1.0, 0.2, 1.0), root->getSpacing(), particle_t::FLUID);
	root->loadParticlesFromFile("../../../Media/toy.obj", particle_t::SAND);

	root->setDissolutionFlag(dissolution);
	root->initSync();

	auto ptRender1 = std::make_shared<PointRenderModule>();
	ptRender1->setColor(Vector3f(1, 0, 1));
	ptRender1->setColorRange(0, 1);
	root->addVisualModule(ptRender1);

	root->m_phase_concentration.connect(&ptRender1->m_vecIndex);

	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(root->getTopologyModule());
		auto& points = pSet->getPoints();
		HostArray<Vector3f> hpoints(points.size());
		Function1Pt::copy(hpoints, points);

		ofstream outf("Particles.txt");
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
}

void CreateSceneCrag(int dissolution)
{
	SceneGraph& scene = SceneGraph::getInstance();

	auto root = scene.createNewScene<FastMultiphaseSPH<DataType3f>>();

	using particle_t = FastMultiphaseSPH<DataType3f>::particle_t;

	root->loadParticlesAABBSurface(Vector3f(-1.02, -0.02, -1.02), Vector3f(1.02, 2.5, 1.02), root->getSpacing(), particle_t::BOUDARY);

	root->loadParticlesAABBVolume(Vector3f(-1.0, 0.0, -1.0), Vector3f(1.0, 0.2, 1.0), root->getSpacing(), particle_t::FLUID);
	root->loadParticlesFromFile("../../../Media/crag.obj", particle_t::SAND);

	root->setDissolutionFlag(dissolution);
	root->initSync();

	auto ptRender1 = std::make_shared<PointRenderModule>();
	ptRender1->setColor(Vector3f(1, 0, 1));
	ptRender1->setColorRange(0, 1);
	root->addVisualModule(ptRender1);

	root->m_phase_concentration.connect(&ptRender1->m_vecIndex);

	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(root->getTopologyModule());
		auto& points = pSet->getPoints();
		HostArray<Vector3f> hpoints(points.size());
		Function1Pt::copy(hpoints, points);

		ofstream outf("Particles.txt");
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
}

int main(int argc, char ** argv)
{
	std::string scene = "block";
	int dissolution = 1;
	if (argc >= 2) scene = argv[1];
	if (argc >= 3) dissolution = atoi(argv[2]);

	if (scene == "block")
		CreateSceneBlock(dissolution);
	else if (scene == "toy")
		CreateSceneToy(dissolution);
	else if (scene == "crag")
		CreateSceneCrag(dissolution);
	else {
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


