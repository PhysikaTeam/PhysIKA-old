/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Simulate viscoplasticity with projective peridynamics
 *               reference <Projective peridynamics for modeling versatile elastoplastic materials>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-16
 * @description: poslish code
 * @version    : 1.1
 */

#include <memory>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Framework/Log.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/RigidBody.h"
#include "Rendering/PointRenderModule.h"
#include "Rendering/RigidMeshRender.h"
#include "GUI/GlutGUI/GLApp.h"

#include "ParticleViscoplasticBody.h"

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

/**
 * setup scene: 2 viscoplastic bunnies fall through 5 bars
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    //scene boundary
    Vector3f                                    lo0  = Vector3f(0);
    Vector3f                                    hi0  = Vector3f(1);
    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(lo0, hi0, 0.005, true);

    //5 bar obstacles
    for (size_t i = 0; i < 5; i++)
    {
        Vector3f lo     = Vector3f(0.2 + i * 0.08, 0.2, 0);
        Vector3f hi     = Vector3f(0.25 + i * 0.08, 0.25, 1);
        Vector3f scale  = (hi - lo) / 2;
        Vector3f center = (hi + lo) / 2;

        root->loadCube(lo, hi, 0.005, false);
        //dummy rigid body for rendering
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

    //first bunny
    std::shared_ptr<ParticleViscoplasticBody<DataType3f>> bunny_0 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
    root->addParticleSystem(bunny_0);
    bunny_0->setMass(1.0);
    bunny_0->loadParticles("../../Media/bunny/bunny_points.obj");
    bunny_0->loadSurface("../../Media/bunny/bunny_mesh.obj");
    bunny_0->translate(Vector3f(0.4, 0.4, 0.5));
    auto ptRender = std::make_shared<PointRenderModule>();
    ptRender->setColor(Vector3f(0, 1, 1));
    bunny_0->addVisualModule(ptRender);

    //second bunny
    std::shared_ptr<ParticleViscoplasticBody<DataType3f>> bunny_1 = std::make_shared<ParticleViscoplasticBody<DataType3f>>();
    root->addParticleSystem(bunny_1);
    bunny_1->setMass(1.0);
    bunny_1->loadParticles("../../Media/bunny/bunny_points.obj");
    bunny_1->loadSurface("../../Media/bunny/bunny_mesh.obj");
    bunny_1->translate(Vector3f(0.4, 0.4, 0.9));
    auto ptRender2 = std::make_shared<PointRenderModule>();
    ptRender2->setColor(Vector3f(1, 0, 1));
    bunny_1->addVisualModule(ptRender2);
}

int main()
{
    createScene();

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
