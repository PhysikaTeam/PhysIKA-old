/**
 * @author     : He YingXiang (heyx0418@163.com)
 * @date       : 2021-03-31
 * @description: demo of discrete collision detection between meshes
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-18
 * @description: poslish code
 * @version    : 1.1
 */

#include <memory>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Dynamics/RigidBody/RigidCollisionBody.h"
#include "Rendering/SurfaceMeshRender.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace std;
using namespace PhysIKA;

std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> CollisionManager::Meshes = {};  //list of surface meshes managed by CollisionManager
std::vector<std::shared_ptr<SurfaceMeshRender>>        SFRender                 = {};  //list of renderer for the bunnies

std::shared_ptr<RigidCollisionBody<DataType3f>> bunny;

//DCD
std::shared_ptr<CollidatableTriangleMesh<DataType3f>> DCD                                        = {};
std::shared_ptr<bvh>                                  CollidatableTriangleMesh<DataType3f>::bvh1 = nullptr;
std::shared_ptr<bvh>                                  CollidatableTriangleMesh<DataType3f>::bvh2 = nullptr;

/**
 * setup scene: 9 bunnies where the one in the center could be moved by the user through keyboard
 * the bunnies are registered to CollisionManager::Meshes and SFRender
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setFrameRate(500);
    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();

    //the one bunny in the center
    bunny = std::make_shared<RigidCollisionBody<DataType3f>>();
    bunny->setMass(1.0);
    root->addRigidBody(bunny);
    bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
    auto sRender = std::make_shared<SurfaceMeshRender>();
    bunny->getSurfaceNode()->addVisualModule(sRender);
    sRender->setColor(Vector3f(0, 1, 0));

    //the other 8 bunnies
    int    idx = 0;
    double dx = 0, dy = 0, dz = 0;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++)
            {
                if (i == 1 && j == 1 && k == 1)
                    continue;

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

/**
 * check collision between the bunnies in scene created by createScene()
 * The bunny that collides with the one controlled by the user is rendered with red color
 */
void checkCollision()
{
    std::vector<int> collisionset;
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

/**
 * keyboard callback
 */
void keyfunc(unsigned char key, int x, int y)
{
    GLApp* window = static_cast<GLApp*>(glutGetWindowData());
    assert(window);
    switch (key)
    {
        case 27:  //ESC: close window
            glutLeaveMainLoop();
            return;
        case 's':  //s: save screen shot
            window->saveScreen();
            break;
        case 'l':  //l: move right
            bunny->translate({ 0.04, 0, 0 });
            break;
        case 'j':  //j: move left
            bunny->translate({ -0.04, 0, 0 });
            break;
        case 'i':  //i: move up
            bunny->translate({ 0, 0.04, 0 });
            break;
        case 'k':  //k: move down
            bunny->translate({ 0, -0.04, 0 });
            break;
        case 'o':  //o: move forward
            bunny->translate({ 0, 0, 0.04 });
            break;
        case 'u':  //u: move backward
            bunny->translate({ 0, 0, -0.04 });
            break;
        case 'r':  //reset
            bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
            break;
    }
    checkCollision();
}

int main()
{
    createScene();

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
