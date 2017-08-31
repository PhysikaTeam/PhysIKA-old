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
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Transform/transform_3d.h"

#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_IO/Image_IO/image_io.h"

#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_Render/Color/color.h"

#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"

using namespace std;
using namespace Physika;

SurfaceMesh<double> ball_mesh;
SurfaceMesh<double> box_mesh;
SurfaceMesh<double> bottom_plane_mesh;
SurfaceMesh<double> teapot_mesh;

Transform<double, 3> box_transform;
Transform<double, 3> teapot_transform;

SurfaceMeshRender<double> ball_mesh_render;
SurfaceMeshRender<double> box_mesh_render;
SurfaceMeshRender<double> bottom_plane_mesh_render;
SurfaceMeshRender<double> teapot_mesh_render;

GlutWindow * glut_window_ptr = NULL;
ScreenBasedRenderManager * screen_based_render_manager_ptr = NULL;


void initFunction()
{
    
    cout << "loading mesh......" << endl;
    ObjMeshIO<double>::load("H:/Physika/Physika_Test/Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/ball_high.obj", &ball_mesh);
    ObjMeshIO<double>::load("H:/Physika/Physika_Test/Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/box_10x.obj", &box_mesh);
    ObjMeshIO<double>::load("H:/Physika/Physika_Test/Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/bottom_plane.obj", &bottom_plane_mesh);
    ObjMeshIO<double>::load("H:/Physika/Physika_Test/Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/teapot.obj", &teapot_mesh);
    cout << "mesh loaded successfully!\n";

    ball_mesh_render.setSurfaceMesh(&ball_mesh);
    ball_mesh_render.enableDisplayList();

    
    box_mesh_render.setSurfaceMesh(&box_mesh);
    box_transform.setTranslation({0, 10, 0});
    box_mesh_render.setTransform(&box_transform);
    box_mesh_render.enableDisplayList();

    bottom_plane_mesh_render.setSurfaceMesh(&bottom_plane_mesh);
    bottom_plane_mesh_render.enableDisplayList();

    teapot_mesh_render.setSurfaceMesh(&teapot_mesh);
    teapot_transform.setScale({ 3,3,3 });
    teapot_transform.setTranslation({ 0, 5, 0 });
    teapot_mesh_render.setTransform(&teapot_transform);
    teapot_mesh_render.enableDisplayList();

    screen_based_render_manager_ptr = new ScreenBasedRenderManager(glut_window_ptr);
    
    screen_based_render_manager_ptr->addPlane(Vector< float, 4>(0.0f, 1.0f, 0.0f, 0.0f));
    screen_based_render_manager_ptr->setLightPos({ 0.0f, 200.0f, -100.0f });
    screen_based_render_manager_ptr->setLightTarget({0.0f, 0.0f, 0.0f});
    screen_based_render_manager_ptr->setLightFov(15.0f);
    screen_based_render_manager_ptr->setLightSpotMin(0.0);
    screen_based_render_manager_ptr->setLightSpotMax(0.5);

    //screen_based_render_manager_ptr->addRender(&box_mesh_render);
    screen_based_render_manager_ptr->addRender(&teapot_mesh_render);
    //screen_based_render_manager_ptr->addRender(&bottom_plane_mesh_render);

}

void displayFunction()
{
    screen_based_render_manager_ptr->render();
}

int main()
{
    GlutWindow glut_window("screen_based_render_manager_test", 1280, 720);
    glut_window_ptr = &glut_window;

    cout << "Window name: " << glut_window.name() << "\n";
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

    glut_window.setCameraPosition(Vector<double, 3>(0, 5, 5));
    glut_window.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    glut_window.setCameraNearClip(0.01);
    glut_window.setCameraFarClip(1.0e3);

    glut_window.setDisplayFunction(displayFunction);
    glut_window.setInitFunction(initFunction);
    glut_window.createWindow();
    return 0;
}