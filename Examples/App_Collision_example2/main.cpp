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
#include "Dynamics/RigidBody/RigidCollisionBody.h"

#include "Framework/Collision/CollidableSDF.h"
#include "Framework/Collision/CollidablePoints.h"
#include "Framework/Collision/CollisionSDF.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include "Framework/Collision/Collision.h"
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
#include "Core/Vector/vector_3d.h"
#include "Framework/Topology/Primitive3D.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Topology/TriangleSet.h"
#include "Dynamics/RigidBody/RigidCollisionBody.h"

using namespace std;
using namespace PhysIKA;


std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> CollisionManager::Meshes = {};
std::vector<std::shared_ptr<SurfaceMeshRender>> SFRender = {};

std::shared_ptr<RigidCollisionBody<DataType3f>> human;
std::shared_ptr<RigidCollisionBody<DataType3f>> cloth;

std::shared_ptr<RigidCollisionBody<DataType3f>> humanCollisionTriangles;
std::shared_ptr<RigidCollisionBody<DataType3f>> clothCollisionTriangles;

std::shared_ptr<StaticBoundary<DataType3f>> root;

auto instance = Collision::getInstance();

void checkCollision(bool update = false);

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setFrameRate(500);
	root = scene.createNewScene<StaticBoundary<DataType3f>>();

	cloth = std::make_shared<RigidCollisionBody<DataType3f>>();
	root->addRigidBody(cloth);
	cloth->loadSurface("../../Media/character/cloth.obj");
	auto clothShader = std::make_shared<SurfaceMeshRender>();
	cloth->getSurfaceNode()->addVisualModule(clothShader);
	clothShader->setColor(Vector3f(1.0f, 0.1f, 0.2f));
	
	human = std::make_shared<RigidCollisionBody<DataType3f>>();
	root->addRigidBody(human);
	human->loadSurface("../../Media/character/body.obj");
	auto humanShader = std::make_shared<SurfaceMeshRender>();
	human->getSurfaceNode()->addVisualModule(humanShader);
	humanShader->setColor(Vector3f(0.8f, 0.8f, 0.8f));

	instance->transformMesh(*cloth->getmeshPtr(), 0);
	instance->transformMesh(*human->getmeshPtr(), 1);

	checkCollision(true);
}

void checkCollision(bool update) {
	instance->collid();
	auto pairs = instance->getContactPairs();

	std::set<int> clothCollisionTriangleIndex;
	std::set<int> humanCollisionTriangleIndex;
	
	int count = 0;
	for (int i = 0; i < pairs.size(); i++) {
		const auto& t1 = pairs[i][0];
		const auto& t2 = pairs[i][1];

		if (t1.id0() == t2.id0()) //self cd
			continue;

		printf("%d: (%d, %d) - (%d, %d)\n", count + 1, t1.id0(), t1.id1(), t2.id0(), t2.id1());

		clothCollisionTriangleIndex.insert(t1.id1());
		humanCollisionTriangleIndex.insert(t2.id1());

		count++;
	}

	printf("Found %d inter-object contacts...\n", count);

	if (!update)
		return;

	// assemble result for display, a really silly way for displaying...
	// cloth 
	{
		clothCollisionTriangles = make_shared<RigidCollisionBody<DataType3f>>();
		root->addRigidBody(clothCollisionTriangles);

		std::vector<DataType3f::Coord> points;
		std::vector<DataType3f::Coord> normals;
		for (const auto index : clothCollisionTriangleIndex) {
			const auto triset = cloth->getmeshPtr()->getTriangleSet();
			auto faceIndices = triset->getHTriangles()[index];
			for (int j = 0; j < 3; ++j) {
				points.push_back(triset->gethPoints()[faceIndices[j]]);
				normals.push_back(triset->gethNormals()[faceIndices[j]]);
			}
		}

		std::vector<TopologyModule::Triangle> triangles;
		for (std::size_t i = 0; i < clothCollisionTriangleIndex.size(); ++i) {
			triangles.push_back(TopologyModule::Triangle(3 * i, 3 * i + 1, 3 * i + 2));
		}

		auto triSet = make_shared<TriangleSet<DataType3f>>();
		triSet->setPoints(points);
		triSet->setNormals(normals);
		triSet->setTriangles(triangles);

		clothCollisionTriangles->getmeshPtr()->loadFromSet(triSet);

		auto shader = std::make_shared<SurfaceMeshRender>();
		clothCollisionTriangles->getSurfaceNode()->addVisualModule(shader);
		shader->setColor(Vector3f(0.8f, 0.8f, 0.8f));
	}

	// human
	{
		humanCollisionTriangles = make_shared<RigidCollisionBody<DataType3f>>();
		root->addRigidBody(humanCollisionTriangles);

		std::vector<DataType3f::Coord> points;
		std::vector<DataType3f::Coord> normals;
		for (const auto index : humanCollisionTriangleIndex) {
			const auto triset = human->getmeshPtr()->getTriangleSet();
			auto faceIndices = triset->getHTriangles()[index];
			for (int j = 0; j < 3; ++j) {
				points.push_back(triset->gethPoints()[faceIndices[j]]);
				normals.push_back(triset->gethNormals()[faceIndices[j]]);
			}
		}

		std::vector<TopologyModule::Triangle> triangles;
		for (std::size_t i = 0; i < humanCollisionTriangleIndex.size(); ++i) {
			triangles.push_back(TopologyModule::Triangle(3 * i, 3 * i + 1, 3 * i + 2));
		}

		auto triSet = make_shared<TriangleSet<DataType3f>>();
		triSet->setPoints(points);
		triSet->setNormals(normals);
		triSet->setTriangles(triangles);

		humanCollisionTriangles->getmeshPtr()->loadFromSet(triSet);

		auto shader = std::make_shared<SurfaceMeshRender>();
		humanCollisionTriangles->getSurfaceNode()->addVisualModule(shader);
		shader->setColor(Vector3f(1.0f, 0.1f, 0.2f));
	}
}

void display(bool showResult) {
	cloth->getSurfaceNode()->setVisible(!showResult);
	human->getSurfaceNode()->setVisible(!showResult);

	if (!instance->getContactPairs().empty()) {
		clothCollisionTriangles->getSurfaceNode()->setVisible(showResult);
		humanCollisionTriangles->getSurfaceNode()->setVisible(showResult);
	}
}

void keyboardFunc(unsigned char key, int x, int y) {
	GLApp* window = static_cast<GLApp*>(glutGetWindowData());
	static bool tPressed = false;
	
	switch (key) {
	case 27: glutLeaveMainLoop(); return;
	case 's': window->saveScreen(); break;
	case 't':
		tPressed = !tPressed;
		display(tPressed);
		break;
	case '2':
		checkCollision();
		break;
	}
}

int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::DebugInfo);
	Log::sendMessage(Log::DebugInfo, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);
	window.setKeyboardFunction(keyboardFunc);

	window.mainLoop();

	Log::sendMessage(Log::DebugInfo, "Simulation end!");
	return 0;
}