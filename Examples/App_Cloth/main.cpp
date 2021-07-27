/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-06
 * @description: Simulate cloth with projective peridynamics
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
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Rendering/PointRenderModule.h"
#include "Rendering/RigidMeshRender.h"
#include "GUI/GlutGUI/GLApp.h"

#include "ParticleCloth.h"

using namespace PhysIKA;

/**
 * Setup scene, multiple cloths fall onto a sphere.
 * Self-collision between cloths is not handled.
 */
void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();

    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(-0.1f, 0.0f, -1.0f), Vector3f(1.1f, 2.0f, 1.1f), 0.02f, true);  //scene boundary
    root->loadShpere(Vector3f(0.5, 0.2f, 0.5f), 0.2f, 0.01f, false);                        //sphere in scene

    {
        //add a dummy rigid body sphere in scene, for rendering
        std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
        root->addRigidBody(rigidbody);
        rigidbody->loadShape("../../Media/standard/standard_sphere.obj");
        rigidbody->setActive(false);  //make sure the rigid body is not simulated
        auto rigidTri = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(rigidbody->getSurface()->getTopologyModule());
        rigidTri->scale(0.2f);
        rigidTri->translate(Vector3f(0.5, 0.2, 0.5));

        auto renderModule = std::make_shared<RigidMeshRender>(rigidbody->getTransformationFrame());
        renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / ( double )1000, 0.8));
        rigidbody->getSurface()->addVisualModule(renderModule);
    }

    int cloth_num = 24;
    for (int i = 0; i < cloth_num; i++)
    {
        std::shared_ptr<ParticleCloth<DataType3f>> child = std::make_shared<ParticleCloth<DataType3f>>();
        root->addParticleSystem(child);
        child->setMass(1.0);
        child->loadParticles("../../Media/cloth/clothLarge.obj");
        child->loadSurface("../../Media/cloth/clothLarge.obj");
        child->translate(Vector3f(0.0f, 0.3f + 0.02 * i, 0.0f));
        child->setVisible(true);

        auto m_pointsRender = std::make_shared<PointRenderModule>();
        m_pointsRender->setColor(Vector3f(1 - 0.04 * i, 0.04 * i, 1));
        child->addVisualModule(m_pointsRender);
    }
}

int main()
{
    createScene();

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();

    return 0;
}
