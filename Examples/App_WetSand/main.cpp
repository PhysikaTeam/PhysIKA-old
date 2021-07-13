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

#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/ParticleSystem/ParticleElastoplasticBody.h"
#include "Dynamics/ParticleSystem/GranularModule.h"

#include "Framework/Topology/TriangleSet.h"
#include "Rendering/RigidMeshRender.h"
#include "Rendering/PointRenderModule.h"

using namespace std;
using namespace PhysIKA;

void RecieveLogMessage(const Log::Message &m)
{
    switch (m.type)
    {
    case Log::Info:
        cout << ">>>: " << m.text << endl;
        break;
    case Log::Warning:
        cout << "???: " << m.text << endl;
        break;
    case Log::Error:
        cout << "!!!: " << m.text << endl;
        break;
    case Log::User:
        cout << ">>>: " << m.text << endl;
        break;
    default:
        break;
    }
}

void CreateScene()
{
    SceneGraph &scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadSDF("../../Media/bar/bar.sdf", false);
    root->translate(Vector3f(0.2f, 0.2f, 0));
    root->loadCube(Vector3f(0), Vector3f(1), 0.005, true);

    std::shared_ptr<ParticleElastoplasticBody<DataType3f>> child3 = std::make_shared<ParticleElastoplasticBody<DataType3f>>();
    root->addParticleSystem(child3);

    auto ptRender = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(0.50, 0.44, 0.38));
    child3->addVisualModule(ptRender);

    child3->setMass(1.0);
    child3->loadParticles("../../Media/bunny/bunny_points.obj");
    child3->loadSurface("../../Media/bunny/bunny_mesh.obj");
    child3->translate(Vector3f(0.3, 0.4, 0.5));
    child3->setDt(0.001);
    auto elasto = std::make_shared<GranularModule<DataType3f>>();
    elasto->enableFullyReconstruction();
    child3->setElastoplasticitySolver(elasto);
    elasto->setCohesion(0.001);

    // Output all particles to .txt file.
    {
        auto pSet = TypeInfo::CastPointerDown<PointSet<DataType3f>>(child3->getTopologyModule());
        auto &points = pSet->getPoints();
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
    rigidbody->loadShape("../../Media/bar/bar.obj");
    rigidbody->setActive(false);
    rigidbody->translate(Vector3f(0.2f, 0.2f, 0));

    {
        auto rigidTri = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(rigidbody->getSurface()->getTopologyModule());
        auto renderModule = std::make_shared<RigidMeshRender>(rigidbody->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
        rigidbody->getSurface()->addVisualModule(renderModule);
    }
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
