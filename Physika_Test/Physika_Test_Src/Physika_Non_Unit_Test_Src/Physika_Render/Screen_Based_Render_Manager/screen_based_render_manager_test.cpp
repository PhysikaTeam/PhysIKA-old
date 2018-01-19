/*
* @file screen_based_render_manager_test.cpp
* @brief Test ScreenBasedRenderManager of Physika.
* @author WeiChen
*
* This file is part of Physika, a versatile physics simulation library.
* Copyright (C) 2013- Physika Group.
*
* This Source Code Form is subject to the terms of the GNU General Public License v2.0.
* If a copy of the GPL was not distributed with this file, you can obtain one at:
* http://www.gnu.org/licenses/gpl-2.0.html
*
*/
#include <iostream>
#include <GL/freeglut.h>

#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Utilities/cuda_math_utilities.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Transform/transform_3d.h"

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/Lights/directional_light.h"
#include "Physika_Render/Lights/point_light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flash_light.h"
#include "Physika_Render/Lights/flex_spot_light.h"

#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"

#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render_util.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_solid_render_task.h"
#include "Physika_Render/Plane_Render/plane_render_task.h"

#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"


using namespace std;
using namespace Physika;

SurfaceMesh<double> ball_mesh;
SurfaceMesh<double> box_mesh;
SurfaceMesh<double> bottom_plane_mesh;
SurfaceMesh<double> teapot_mesh;
SurfaceMesh<double> antique_mesh;

Transform<float, 3> box_transform;
Transform<float, 3> teapot_transform;
Transform<float, 3> antique_transform;


void initFunction()
{
    
    cout << "loading mesh......" << endl;
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/ball_high.obj", &ball_mesh);
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/box_10x.obj", &box_mesh);
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/bottom_plane.obj", &bottom_plane_mesh);
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/teapot.obj", &teapot_mesh);
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/scene_dense_mesh_refine_texture.obj", &antique_mesh);
    cout << "mesh loaded successfully!\n";

    //-----------------------------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto teatpot_render_util = make_shared<SurfaceMeshRenderUtil<double>>(&teapot_mesh);
    auto teapot_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(teatpot_render_util);

    //teapot_render_task->disableUseMaterial();
    //teapot_render_task->disableUseTexture();

    //teapot_render_task->enableUseSolidColor();
    //teapot_render_task->setSolidColor(Color4f::Cyan());

    teapot_transform.setScale({ 3,3,3 });
    teapot_transform.setTranslation({ 0, 2, 0 });
    teapot_render_task->setTransform(teapot_transform);

    render_scene_config.pushBackRenderTask(teapot_render_task);

    //-----------------------------------------------------------------------------------------------------------------------------------------
    /*
    auto antique_render_util = make_shared<SurfaceMeshRenderUtil<double>>(&antique_mesh);
    auto antique_render_task = make_shared<SurfaceMeshSolidRenderTask<double>>(antique_render_util);

    antique_transform.setScale({5.0f, 5.0f, 5.0f});
    antique_transform.setRotation({0.0, 0.0, 1.0}, PI / 2.0);
    antique_transform.setTranslation({ 0, 5, 0 });
    antique_render_task->setTransform(antique_transform);

    antique_render_task->disableBindShader();

    render_scene_config.pushBackRenderTask(antique_render_task);
    */

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto plane_render_task = make_shared<PlaneRenderTask>();
    plane_render_task->addPlane({ 0.0f, 1.0f, 0.0f, 0.0f });

    render_scene_config.pushBackRenderTask(plane_render_task);

    //---------------------------------------------------------------------------------------------------------------------
    //light config
    
    auto spot_light = make_shared<SpotLight>();
    spot_light->setPos({ 0, 50, 0 });
    spot_light->setSpotDirection({ 0, -1, 0 });
    spot_light->setSpotCutoff(0.0);
    spot_light->setSpotOuterCutoff(20);
    spot_light->setAmbient(Color4f::Gray());
    render_scene_config.pushBackLight(spot_light);
    

    auto flash_light = make_shared<FlashLight>();
    //flash_light->setAmbient(Color4f::White());
    render_scene_config.pushBackLight(flash_light);

    /*
    auto flex_spot_light = make_shared<FlexSpotLight>();
    flex_spot_light->setPos({100, 50, 0 });
    flex_spot_light->setTarget({ 100, 0, 0 });
    flex_spot_light->setFov(30.f);
    flex_spot_light->setSpotMin(0.6);
    flex_spot_light->setSpotMax(0.8);
    render_scene_config.pushBackLight(flex_spot_light);
    */
    //----------------------------------------------------------------------------------------------------------------------------------------
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glClearDepth(1.0);
    glClearColor(0.49, 0.49, 0.49, 1.0);

    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
}

void displayFunction()
{
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();

    //render_manager.disableUseShadowmap();
    render_manager.disableUseGammaCorrection();
    //render_manager.enableUseHDR();


    render_manager.render();

    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
    cur_window->displayFrameRate();

    glutPostRedisplay();
    glutSwapBuffers();
}

int main()
{
    GlutWindow glut_window("screen_based_render_manager_test", 1280, 720);

    cout << "Window name: " << glut_window.name() << "\n";
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    render_scene_config.setCameraPosition(Vector<double, 3>(0, 0, 200));
    render_scene_config.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    render_scene_config.setCameraNearClip(0.001);
    render_scene_config.setCameraFarClip(1.0e4);

    glut_window.setDisplayFunction(displayFunction);
    glut_window.setInitFunction(initFunction);
    glut_window.createWindow();
    return 0;
}