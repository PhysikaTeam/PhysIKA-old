/*
* @file triangle_render_task_test.cpp
* @brief Test triangle render task of Physika.
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
#include <memory>
#include <GL/freeglut.h>

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_Geometry/Boundary_Meshes/face_group.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/ColorBar/ColorMap/color_map.h"

#include "Physika_Render/Lights/directional_light.h"
#include "Physika_Render/Lights/point_light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flash_light.h"


#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Triangle_Render/triangle_render_util.h"
#include "Physika_Render/Triangle_Render/triangle_wireframe_render_task.h"
#include "Physika_Render/Triangle_Render/triangle_solid_render_task.h"
#include "Physika_Render/Triangle_Render/triangle_gl_cuda_buffer.h"
#include "Physika_Render/Utilities/gl_cuda_buffer_test_tool.h"
#include <cuda_runtime_api.h>

using namespace std;
using namespace Physika;

using BoundaryMeshInternal::Vertex;
using SurfaceMeshInternal::Face;
using SurfaceMeshInternal::FaceGroup;
using BoundaryMeshInternal::Material;

SurfaceMesh<double> mesh;

std::vector<Vector3d> getSurfaceMeshTriangles(SurfaceMesh<double> & mesh)
{
    std::vector<Vector3d> pos_vec;

    unsigned int group_num = mesh.numGroups();
    for (unsigned int group_idx = 0; group_idx < group_num; ++group_idx)
    {
        FaceGroup<double> & group_ref = mesh.group(group_idx);
        unsigned int face_num = group_ref.numFaces();

        for (unsigned int face_idx = 0; face_idx < face_num; ++face_idx)
        {
            Face<double> &face_ref = group_ref.face(face_idx);          // get face reference
            unsigned int vertex_num = face_ref.numVertices();

            unsigned int triangle_num = vertex_num - 2;

            for (unsigned int triangle_id = 0; triangle_id < triangle_num; ++triangle_id)
            {
                unsigned int triangle_vert_ids[3] = { 0, triangle_id + 1, triangle_id + 2 };

                unsigned fir_position_ID = face_ref.vertex(triangle_vert_ids[0]).positionIndex();
                const Vector<double, 3> & fir_pos = mesh.vertexPosition(fir_position_ID);

                unsigned sec_position_ID = face_ref.vertex(triangle_vert_ids[1]).positionIndex();
                const Vector<double, 3> & sec_pos = mesh.vertexPosition(sec_position_ID);

                unsigned third_position_ID = face_ref.vertex(triangle_vert_ids[2]).positionIndex();
                const Vector<double, 3> & third_pos = mesh.vertexPosition(third_position_ID);

                pos_vec.push_back(fir_pos);
                pos_vec.push_back(sec_pos);
                pos_vec.push_back(third_pos);
            }
        }
    }

    return pos_vec;
}

void initFunction()
{
    cout << "loading mesh ......" << endl;
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/scene_dense_mesh_refine_texture.obj", &mesh);
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/teapot.obj", &mesh);
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/bottom_plane.obj", &mesh);
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/ball_high.obj", &mesh);
    //ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/SunFlower.obj", &mesh);
    cout << "load mesh successfully" << endl;

    //---------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    //---------------------------------------------------------------------------------------------------------------------
    std::vector<Vector3d>  pos_vec = getSurfaceMeshTriangles(mesh);

    auto triangle_render_util = make_shared<TriangleRenderUtil>();
    //triangle_render_util->setTriangles(pos_vec);

    vector<Vector3f> float_pos_vec;
    for (unsigned int i = 0; i < pos_vec.size(); ++i)
        float_pos_vec.push_back(Vector3f(pos_vec[i][0], pos_vec[i][1], pos_vec[i][2]));

    TriangleGLCudaBuffer triangle_gl_cuda_buffer = triangle_render_util->mapTriangleGLCudaBuffer(float_pos_vec.size() / 3);
    //cudaMemcpy(triangle_gl_cuda_buffer.getCudaPosPtr(), float_pos_vec.data(), 3 * sizeof(float) * float_pos_vec.size(), cudaMemcpyHostToDevice);
    setTriangleGLCudaBuffer(float_pos_vec, triangle_gl_cuda_buffer);
    triangle_render_util->unmapTriangleGLCudaBuffer();


    //---------------------------------------------------------------------------------------------------------------------
    auto triangle_wireframe_render_task = make_shared<TriangleWireframeRenderTask>(triangle_render_util);
    render_scene_config.pushBackRenderTask(triangle_wireframe_render_task);

    //---------------------------------------------------------------------------------------------------------------------
    auto triangle_solid_render_task = make_shared<TriangleSolidRenderTask>(triangle_render_util);
    
    ColorMap<float> color_map;
    vector<Color4f> face_col_vec;
    for (int i = 0; i < triangle_render_util->triangleNum(); ++i)
        face_col_vec.push_back(color_map.colorViaRatio(float(i) / triangle_render_util->triangleNum()));

    triangle_solid_render_task->setTriangleColors(face_col_vec);
    //render_scene_config.pushBackRenderTask(triangle_solid_render_task);

    //---------------------------------------------------------------------------------------------------------------------
    //light config

    //auto directional_light = make_shared<DirectionalLight>();
    //directional_light->setDirection({ 0, -1, 0 });
    //render_scene_config.pushBackLight(directional_light);


    //auto point_light = make_shared<PointLight>();
    ////point_light->setPos({ 0, 10, 0 });
    //point_light->setPos({ 0, 0, 100 });
    ////point_light->setDiffuse(Color4f::Red());
    ////point_light->setSpecular(Color4f::Black());
    //render_scene_config.pushBackLight(point_light);

    //auto spot_light = make_shared<SpotLight>();
    //spot_light->setPos({ 0, 100, 0 });
    //spot_light->setSpotDirection({ 0, -1, 0 });
    //spot_light->setSpotCutoff(0.0);
    //spot_light->setSpotOuterCutoff(20);
    ////spot_light->setSpotExponent(16);
    //spot_light->setAmbient(Color4f::Gray());
    //render_scene_config.pushBackLight(spot_light);

    auto flash_light = make_shared<FlashLight>();
    //flash_light->setAmbient(Color4f::White());
    render_scene_config.pushBackLight(flash_light);

    /*
    auto flash_light_2 = make_shared<FlashLight>();
    flash_light_2->setAmbient(Color4f::White());
    render_scene_config.pushBackLight(flash_light_2);
    */

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glClearDepth(1.0);
    glClearColor(0.49, 0.49, 0.49, 1.0);

    glEnable(GL_DEPTH_TEST);
    //glEnable(GL_CULL_FACE);
}

void displayFunction()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_POLYGON_OFFSET_FILL);
    glPolygonOffset(1.0, 1.0);

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    render_scene_config.renderAllTasks();

    GlutWindow * cur_window = (GlutWindow*)glutGetWindowData();
    cur_window->displayFrameRate();
    glutPostRedisplay();
    glutSwapBuffers();
}

void keyboardFunction(unsigned char key, int x, int y)
{
    GlutWindow::bindDefaultKeys(key, x, y);
    switch (key)
    {
    case 't':
        cout << "test\n";
        break;
    default:
        break;
    }
}

int main()
{
    GlutWindow glut_window;
    cout << "Window name: " << glut_window.name() << "\n";
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    //render_scene_config.setCameraPosition(Vector<double, 3>(0, 0, 200));
    //render_scene_config.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    render_scene_config.setCameraNearClip(0.1);
    render_scene_config.setCameraFarClip(1.0e3);

    glut_window.setDisplayFunction(displayFunction);
    glut_window.setInitFunction(initFunction);

    cout << "Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";
    cout << "Test window with GLUI controls:\n";

    return 0;
}