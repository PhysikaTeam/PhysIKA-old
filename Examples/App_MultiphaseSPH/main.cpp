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
}

int main()
{
	CreateScene();

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


