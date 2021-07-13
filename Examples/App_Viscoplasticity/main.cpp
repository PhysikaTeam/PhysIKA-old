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

#include "Rendering/PointRenderModule.h"

#include "ParticleViscoplasticBody.h"

#include "Rendering/PointRenderModule.h"

#include "Framework/Topology/TriangleSet.h"
#include "Rendering/RigidMeshRender.h"

using namespace std;
using namespace PhysIKA;

void RecieveLogMessage(const Log::Message& m)
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
    SceneGraph& scene = SceneGraph::getInstance();

    Vector3f lo0 = Vector3f(0);
    Vector3f hi0 = Vector3f(1);

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(lo0, hi0, 0.005, true);

    //Vector3f scale = (hi0 - lo0) / 2;
    //Vector3f center = (hi0 + lo0) / 2;
    //std::shared_ptr<RigidBody<DataType3f>> rigidbody0 = std::make_shared<RigidBody<DataType3f>>();
    //root->addRigidBody(rigidbody0);
    //rigidbody0->loadShape("../../Media/standard/standard_cube.obj");
    //rigidbody0->setActive(false);
    //auto rigidTri0 = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(rigidbody0->getSurface()->getTopologyModule());
    //rigidTri0->scale(scale);
    //rigidTri0->translate(center);

    //auto renderModule = std::make_shared<RigidMeshRender>(rigidbody0->getTransformationFrame());
    //renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
    //rigidbody0->getSurface()->addVisualModule(renderModule);

    for (size_t i = 0; i < 5; i++)
    {
        Vector3f lo     = Vector3f(0.2 + i * 0.08, 0.2, 0);
        Vector3f hi     = Vector3f(0.25 + i * 0.08, 0.25, 1);
        Vector3f scale  = (hi - lo) / 2;
        Vector3f center = (hi + lo) / 2;

        root->loadCube(lo, hi, 0.005, false, true);

        std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
        root->addRigidBody(rigidbody);
        rigidbody->loadShape("../../Media/standard/standard_cube.obj");
        rigidbody->setActive(false);
        auto rigidTri = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(rigidbody->getSurface()->getTopologyModule());
        rigidTri->scale(scale);
        rigidTri->translate(center);

        auto renderModule = std::make_shared<RigidMeshRender>(rigidbody->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        rigidbody->getSurface()->addVisualModule(renderModule);
    }

    std::shared_ptr<ParticleViscoplasticBody<DataType3f>> child3 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
    root->addParticleSystem(child3);

    auto ptRender = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(0, 1, 1));
    child3->addVisualModule(ptRender);

    child3->setMass(1.0);
    child3->loadParticles("../../Media/bunny/bunny_points.obj");
    child3->loadSurface("../../Media/bunny/bunny_mesh.obj");
    child3->translate(Vector3f(0.4, 0.4, 0.5));

    // Output all particles to .txt file.
    {
        auto                pSet   = TypeInfo::CastPointerDown<PointSet<DataType3f>>(child3->getTopologyModule());
        auto&               points = pSet->getPoints();
        HostArray<Vector3f> hpoints(points.size());
        Function1Pt::copy(hpoints, points);

        ofstream outf("Particles.obj");
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

    std::shared_ptr<ParticleViscoplasticBody<DataType3f>> child4 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
    root->addParticleSystem(child4);
    auto ptRender2 = std::make_shared<PointRenderModule>();
    ptRender2->setColor(Vector3f(1, 0, 1));
    child4->addVisualModule(ptRender2);

    child4->setMass(1.0);
    child4->loadParticles("../../Media/bunny/bunny_points.obj");
    child4->loadSurface("../../Media/bunny/bunny_mesh.obj");
    child4->translate(Vector3f(0.4, 0.4, 0.9));

    // Output all particles to .txt file.
    {
        auto                pSet   = TypeInfo::CastPointerDown<PointSet<DataType3f>>(child3->getTopologyModule());
        auto&               points = pSet->getPoints();
        HostArray<Vector3f> hpoints(points.size());
        Function1Pt::copy(hpoints, points);

        ofstream outf("Particles.obj", ios::app);
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
