#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Physika_GUI/GlutGUI/GLApp.h"

#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Log.h"

#include "Physika_Render/PointRenderModule.h"

#include "Physika_Dynamics/ParticleSystem/ParticleFluid.h"
#include "Physika_Dynamics/ParticleSystem/Peridynamics.h"

#include "Physika_Framework/Collision/CollidableSDF.h"
#include "Physika_Framework/Collision/CollidablePoints.h"
#include "Physika_Framework/Collision/CollisionSDF.h"
#include "Physika_Dynamics/RigidBody/RigidBody.h"
#include "Physika_Framework/Framework/Gravity.h"
#include "Physika_Dynamics/ParticleSystem/FixedPoints.h"
#include "Physika_Framework/Collision/CollisionPoints.h"

using namespace std;
using namespace Physika;

#include "Physika_GUI/QtGUI/QtApp.h"

int main()
{
	SceneGraph& scene = SceneGraph::getInstance();

	//create root node
	std::shared_ptr<Node> root = scene.createNewScene<Node>("root");

	auto pSet = std::make_shared<PointSet<DataType3f>>();
	root->setTopologyModule(pSet);

	std::vector<Vector3f> positions;
	for (float x = 0.4; x < 0.6; x += 0.005f) {
		for (float y = 0.1; y < 0.1075; y += 0.005f) {
			for (float z = 0.4; z < 0.6; z += 0.005f) {
				positions.push_back(Vector3f(Real(x), Real(y), Real(z)));
			}
		}
	}
	pSet->setPoints(positions);

	auto pS1 = std::make_shared<Peridynamics<DataType3f>>();
	root->setNumericalModel(pS1);

	auto render = std::make_shared<PointRenderModule>();
	render->scale(100.0f, 100.0f, 100.0f);
	root->addVisualModule(render);

	QtApp app;
	app.createWindow(1024, 768);

	app.mainLoop();
}
