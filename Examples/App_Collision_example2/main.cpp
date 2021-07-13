#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Core/Vector/vector_3d.h"
#include "Framework/Framework/SceneGraph.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include "Framework/Collision/Collision.h"
#include "Framework/Collision/CollisionPoints.h"
#include "Framework/Topology/Primitive3D.h"
#include "Framework/Topology/TriangleSet.h"
#include "Dynamics/RigidBody/RigidCollisionBody.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Rendering/SurfaceMeshRender.h"

#include "SurfaceLineRender.h"

using namespace std;
using namespace PhysIKA;

std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> CollisionManager::Meshes = {};

std::shared_ptr<RigidCollisionBody<DataType3f>> human;
std::shared_ptr<RigidCollisionBody<DataType3f>> cloth;

std::shared_ptr<RigidCollisionBody<DataType3f>> humanCollisionTriangles;
std::shared_ptr<RigidCollisionBody<DataType3f>> clothCollisionTriangles;

std::shared_ptr<StaticBoundary<DataType3f>> root;

enum class ExampleMode
{
    Static,
    Dynamic
};
ExampleMode mode = ExampleMode::Static;

bool pause = true;

auto instance = Collision::getInstance();

void checkCollision();

void CreateScene(const std::string& clothPath, const std::string& humanPath)
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.invalid();
    scene.setFrameRate(500);
    root = scene.createNewScene<StaticBoundary<DataType3f>>();

    cloth = std::make_shared<RigidCollisionBody<DataType3f>>();
    root->addRigidBody(cloth);
    cloth->loadSurface(clothPath);
    auto clothShader = std::make_shared<SurfaceLineRender>();
    cloth->getSurfaceNode()->addVisualModule(clothShader);
    clothShader->setColor(Vector3f(1.0f, 0.1f, 0.2f));

    human = std::make_shared<RigidCollisionBody<DataType3f>>();
    root->addRigidBody(human);
    human->loadSurface(humanPath);
    auto humanShader = std::make_shared<SurfaceLineRender>();
    human->getSurfaceNode()->addVisualModule(humanShader);
    humanShader->setColor(Vector3f(0.8f, 0.8f, 0.8f));

    instance->transformMesh(*cloth->getmeshPtr(), 0);
    instance->transformMesh(*human->getmeshPtr(), 1);

    checkCollision();
    scene.initialize();
}

void checkCollision()
{
    instance->collid();
    auto pairs = instance->getContactPairs();

    std::set<int> clothCollisionTriangleIndex;
    std::set<int> humanCollisionTriangleIndex;

    int count = 0;
    for (int i = 0; i < pairs.size(); i++)
    {
        const auto& t1 = pairs[i][0];
        const auto& t2 = pairs[i][1];

        if (t1.id0() == t2.id0())  //self cd
            continue;

        printf("%d: (%d, %d) - (%d, %d)\n", count + 1, t1.id0(), t1.id1(), t2.id0(), t2.id1());

        clothCollisionTriangleIndex.insert(t1.id1());
        humanCollisionTriangleIndex.insert(t2.id1());

        count++;
    }

    printf("Found %d inter-object contacts...\n", count);

    // assemble result for display, a really silly way for displaying...
    // cloth
    {
        clothCollisionTriangles = make_shared<RigidCollisionBody<DataType3f>>();
        root->addRigidBody(clothCollisionTriangles);

        std::vector<DataType3f::Coord> points;
        std::vector<DataType3f::Coord> normals;
        for (const auto index : clothCollisionTriangleIndex)
        {
            const auto triset      = cloth->getmeshPtr()->getTriangleSet();
            auto       faceIndices = triset->getHTriangles()[index];
            for (int j = 0; j < 3; ++j)
            {
                points.push_back(triset->gethPoints()[faceIndices[j]]);
                normals.push_back(triset->gethNormals()[faceIndices[j]]);
            }
        }

        std::vector<TopologyModule::Triangle> triangles;
        for (std::size_t i = 0; i < clothCollisionTriangleIndex.size(); ++i)
        {
            triangles.push_back(TopologyModule::Triangle(3 * i, 3 * i + 1, 3 * i + 2));
        }

        auto triSet = make_shared<TriangleSet<DataType3f>>();
        triSet->setPoints(points);
        triSet->setNormals(normals);
        triSet->setTriangles(triangles);

        clothCollisionTriangles->getmeshPtr()->loadFromSet(triSet);

        auto shader = std::make_shared<SurfaceMeshRender>();
        clothCollisionTriangles->getSurfaceNode()->addVisualModule(shader);
        shader->setColor(Vector3f(0.0f, 1.0f, 0.0f));
    }

    // human
    {
        humanCollisionTriangles = make_shared<RigidCollisionBody<DataType3f>>();
        root->addRigidBody(humanCollisionTriangles);

        std::vector<DataType3f::Coord> points;
        std::vector<DataType3f::Coord> normals;
        for (const auto index : humanCollisionTriangleIndex)
        {
            const auto triset      = human->getmeshPtr()->getTriangleSet();
            auto       faceIndices = triset->getHTriangles()[index];
            for (int j = 0; j < 3; ++j)
            {
                points.push_back(triset->gethPoints()[faceIndices[j]]);
                normals.push_back(triset->gethNormals()[faceIndices[j]]);
            }
        }

        std::vector<TopologyModule::Triangle> triangles;
        for (std::size_t i = 0; i < humanCollisionTriangleIndex.size(); ++i)
        {
            triangles.push_back(TopologyModule::Triangle(3 * i, 3 * i + 1, 3 * i + 2));
        }

        auto triSet = make_shared<TriangleSet<DataType3f>>();
        triSet->setPoints(points);
        triSet->setNormals(normals);
        triSet->setTriangles(triangles);

        humanCollisionTriangles->getmeshPtr()->loadFromSet(triSet);

        auto shader = std::make_shared<SurfaceMeshRender>();
        humanCollisionTriangles->getSurfaceNode()->addVisualModule(shader);
        shader->setColor(Vector3f(0.0f, 1.0f, 0.0f));
    }
}

void keyboardFunc(unsigned char key, int x, int y)
{
    GLApp* window = static_cast<GLApp*>(glutGetWindowData());

    switch (key)
    {
        case 27:
            glutLeaveMainLoop();
            return;
        case 's':
            window->saveScreen();
            break;
        case ' ':
            pause = !pause;
            break;
    }
}

void idleFunction()
{
    static int index = 0;
    if (index > 93)
        return;

    if (mode == ExampleMode::Static)
    {
        CreateScene("../../Media/character/cloth000-uv.obj", "../../Media/character/pose000.obj");
        index = 94;
        return;
    }

    if ((pause && index > 0) || index > 93)
        return;

    std::stringstream clothPath, humanPath;
    clothPath << "../../Media/bishop/cloth" << setw(3) << setfill('0') << index << "-uv.obj";
    humanPath << "../../Media/bishop/pose" << setw(3) << setfill('0') << index << ".obj";

    CreateScene(clothPath.str(), humanPath.str());

    index++;
}

int main(int argc, char* argv[])
{
    if (argc == 2)
    {
        if (std::string(argv[1]) == "-s")
        {
            mode = ExampleMode::Static;
        }
        else if (std::string(argv[1]) == "-d")
        {
            mode = ExampleMode::Dynamic;
        }
    }

    GLApp window;
    window.setIdleFunction(idleFunction);
    if (mode == ExampleMode::Dynamic)
    {
        window.setKeyboardFunction(keyboardFunc);
    }

    window.createWindow(1024, 768);
    window.mainLoop();

    return 0;
}