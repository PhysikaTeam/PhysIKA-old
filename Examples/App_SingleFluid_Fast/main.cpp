#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"

#include "Rendering/PointRenderModule.h"
#include "Rendering/SurfaceMeshRender.h"

#include "Dynamics/ParticleSystem/ParticleFluidFast.h"

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
	scene.setUpperBound(Vector3f(1.5, 1, 1.5));
	scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

	std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
	root->loadCube(Vector3f(-0.5, 0, -0.5), Vector3f(1.5, 2, 1.5), 0.02, true);
	root->loadSDF("../../Media/bowl/bowl.sdf", false);

	std::shared_ptr<ParticleFluidFast<DataType3f>> child1 = std::make_shared<ParticleFluidFast<DataType3f>>();
	root->addParticleSystem(child1);

	auto ptRender = std::make_shared<PointRenderModule>();
	ptRender->setColor(Vector3f(1, 0, 0));
	ptRender->setColorRange(0, 4);
	child1->addVisualModule(ptRender);

	//child1->loadParticles("../Media/fluid/fluid_point.obj");
	child1->loadParticles(Vector3f(0.3, 0.2, 0.4), Vector3f(0.7, 1.5, 0.6475), 0.005);
	child1->setMass(100);
	child1->currentVelocity()->connect(&ptRender->m_vecIndex);

	// Output all particles to .txt file.
	{
		auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(child1->getTopologyModule());
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

	std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	root->addRigidBody(rigidbody);
	rigidbody->loadShape("../../Media/bowl/bowl.obj");
	rigidbody->setActive(false);
	rigidbody->setVisible(true);

	auto sRender2 = std::make_shared<SurfaceMeshRender>();
	sRender2->setColor(Vector3f(0.8f, 0.8f, 0.8f));
	rigidbody->getSurface()->addVisualModule(sRender2);
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


