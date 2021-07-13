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
#include "Framework/Collision/CollidableTriangleMesh.h"
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
#include <memory>
#include "Dynamics/RigidBody/RigidCollisionBody.h"
#include "helper.h"

using namespace std;
using namespace PhysIKA;

std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> CollisionManager::Meshes = {};
std::vector<std::shared_ptr<SurfaceMeshRender>>        SFRender                 = {};

std::shared_ptr<RigidCollisionBody<DataType3f>> bunny;

//DCD
std::shared_ptr<CollidatableTriangleMesh<DataType3f>> DCD                                        = {};
std::shared_ptr<bvh>                                  CollidatableTriangleMesh<DataType3f>::bvh1 = nullptr;
std::shared_ptr<bvh>                                  CollidatableTriangleMesh<DataType3f>::bvh2 = nullptr;

void CreateScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setFrameRate(500);
    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();

    bunny =
        std::make_shared<RigidCollisionBody<DataType3f>>();
    bunny->setMass(1.0);
    root->addRigidBody(bunny);
    bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(0, 1, 0));

    int    idx = 0;
    double dx = 0, dy = 0, dz = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)

            {
                if (i == 1 && j == 1 && k == 1)
                    continue;
                //Vector3f rot = randDir<_REAL>();
                //_REAL theta = randDegree<_REAL>(); // 30.0;

                auto nbunny = std::make_shared<RigidCollisionBody<DataType3f>>();
                nbunny->setMass(1.0);
                root->addRigidBody(nbunny);
                nbunny->getmeshPtr()->loadFromSet(TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(bunny->getSurfaceNode()->getTopologyModule()));
                double gap = 0.4;
                nbunny->translate(Vector3f(i * gap - gap, j * gap - gap, k * gap - gap));
                nbunny->postprocess();
                auto sRender = std::make_shared<SurfaceMeshRender>();
                nbunny->getSurfaceNode()->addVisualModule(sRender);
                sRender->setColor(Vector3f(1, 1, 1));

                CollisionManager::Meshes.push_back(nbunny->getmeshPtr());
                SFRender.push_back(sRender);
            }
}

std::vector<int> collisionset;
void             checkCollision()
{
    collisionset.clear();
    for (int i = 0; i < CollisionManager::Meshes.size(); ++i)
    {
        auto cd = DCD->checkCollision(bunny->getmeshPtr(), CollisionManager::Meshes[i]);
        if (cd)
        {
            printf("found collision with %d\n", i);
            collisionset.push_back(i);
        }
        else
        {
            SFRender[i]->setColor({ 1, 1, 1 });
        }
    }
    for (int i = 0; i < collisionset.size(); ++i)
    {
        SFRender[collisionset[i]]->setColor({ 1, 0, 0 });
    }
}

int main()
{
    CreateScene();

    printf("Usage, see helper.h\n\n");
    Log::setOutput("console_log.txt");
    Log::setLevel(Log::DebugInfo);
    Log::sendMessage(Log::DebugInfo, "Simulation begin");

    GLApp window;
    window.setKeyboardFunction(keyfunc);
    window.createWindow(1024, 768);

    window.mainLoop();

    Log::sendMessage(Log::DebugInfo, "Simulation end!");
    return 0;
}
