/**
 * @author     : Chang Yue (Changy1506@buaa.edu.cn)
 * @date       : 2021-05-26
 * @description: dambreak scene with oil rig
 *               demo of paper <Semi-analytical Solid Boundary Conditions for Free Surface Flows>
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-14
 * @description: poslish code
 * @version    : 1.1
 */

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Topology/TriangleSet.h"
#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/ParticleSystem/ParticleFluid.h"
#include "Dynamics/ParticleSystem/SemiAnalyticalSFINode.h"
#include "Dynamics/ParticleSystem/TriangularSurfaceMeshNode.h"
#include "Rendering/SurfaceMeshRender.h"
#include "Rendering/PointRenderModule.h"
#include "GUI/GlutGUI/GLApp.h"

using namespace PhysIKA;

void createScene()
{
    SceneGraph& scene = SceneGraph::getInstance();
    scene.setTotalTime(3.0f);
    scene.setGravity(Vector3f(0.0f, -9.8f, 0.0f));
    scene.setLowerBound(Vector3f(-1.0f, 0.0f, 0.0f));
    scene.setUpperBound(Vector3f(1.0f, 1.0f, 1.0f));
    scene.setFrameRate(100);
    std::shared_ptr<StaticBoundary<DataType3f>> root = scene.createNewScene<StaticBoundary<DataType3f>>();
    root->loadCube(Vector3f(0.01), Vector3f(0.99), 0.01f, true);

    //Particle fluid node
    std::shared_ptr<ParticleFluid<DataType3f>> fluid = std::make_shared<ParticleFluid<DataType3f>>("fluid");
    fluid->loadParticles(Vector3f(-0.985, 0.015, 0.015), Vector3f(-0.585f, 0.6f, 0.985), 0.005);
    auto mf_pointsRender = std::make_shared<PointRenderModule>();
    fluid->addVisualModule(mf_pointsRender);
    mf_pointsRender->setColor(Vector3f(0, 1, 1));
    mf_pointsRender->setColorRange(0, 3);

    //Platform
    auto platform = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("platform");
    platform->getTriangleSet()->loadObjFile("../../Media/platform/spp2.obj");
    platform->scale(1 / 1000.0);
    platform->translate(Vector3f(0.5, 0.0, 0.5));
    auto sRenderf = std::make_shared<SurfaceMeshRender>();
    platform->addVisualModule(sRenderf);
    sRenderf->setColor(Vector3f(1, 1, 0));
    sRenderf->setVisible(true);

    //Scene boundary
    auto boundary = std::make_shared<TriangularSurfaceMeshNode<DataType3f>>("boundary");
    boundary->getTriangleSet()->loadObjFile("../../Media/standard/standard_cube2.obj");
    auto sRenderf2 = std::make_shared<SurfaceMeshRender>();
    boundary->addVisualModule(sRenderf2);
    sRenderf2->setColor(Vector3f(1, 1, 0));
    sRenderf2->setVisible(false);

    //SFI node
    std::shared_ptr<SemiAnalyticalSFINode<DataType3f>> sfi = std::make_shared<SemiAnalyticalSFINode<DataType3f>>();
    sfi->setInteractionDistance(0.016);
    sfi->getParticleVelocity()->connect(&mf_pointsRender->m_vecIndex);
    sfi->addParticleSystem(fluid);
    sfi->addTriangularSurfaceMeshNode(platform);
    sfi->addTriangularSurfaceMeshNode(boundary);
    root->addChild(sfi);

    GLApp window;
    window.createWindow(1024, 768);
    window.mainLoop();
}

int main()
{
    createScene();
    return 0;
}
