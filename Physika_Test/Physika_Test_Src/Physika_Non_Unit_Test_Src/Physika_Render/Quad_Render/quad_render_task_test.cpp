/*
* @file quad_render_task_test.cpp
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
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/ColorBar/ColorMap/color_map.h"

#include "Physika_Render/Lights/directional_light.h"
#include "Physika_Render/Lights/point_light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flash_light.h"


#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Quad_Render/quad_render_util.h"
#include "Physika_Render/Quad_Render/quad_wireframe_render_task.h"
#include "Physika_Render/Quad_Render/quad_solid_render_task.h"

using namespace std;
using namespace Physika;

CubicMesh<double> * mesh;

std::vector<Vector3d> getVolumetricMeshQuads(CubicMesh<double> * mesh)
{
    std::vector<Vector3d> pos_vec;

    unsigned int num_ele = mesh->eleNum();
    for (unsigned int ele_idx = 0; ele_idx < num_ele; ele_idx++)
    {
        Vector<double, 3> pos_0 = mesh->eleVertPos(ele_idx, 0);
        Vector<double, 3> pos_1 = mesh->eleVertPos(ele_idx, 1);
        Vector<double, 3> pos_2 = mesh->eleVertPos(ele_idx, 2);
        Vector<double, 3> pos_3 = mesh->eleVertPos(ele_idx, 3);
        Vector<double, 3> pos_4 = mesh->eleVertPos(ele_idx, 4);
        Vector<double, 3> pos_5 = mesh->eleVertPos(ele_idx, 5);
        Vector<double, 3> pos_6 = mesh->eleVertPos(ele_idx, 6);
        Vector<double, 3> pos_7 = mesh->eleVertPos(ele_idx, 7);

        pos_vec.push_back(pos_0);
        pos_vec.push_back(pos_3);
        pos_vec.push_back(pos_2);
        pos_vec.push_back(pos_1);

        pos_vec.push_back(pos_4);
        pos_vec.push_back(pos_5);
        pos_vec.push_back(pos_6);
        pos_vec.push_back(pos_7);

        pos_vec.push_back(pos_0);
        pos_vec.push_back(pos_1);
        pos_vec.push_back(pos_5);
        pos_vec.push_back(pos_4);

        pos_vec.push_back(pos_3);
        pos_vec.push_back(pos_7);
        pos_vec.push_back(pos_6);
        pos_vec.push_back(pos_2);

        pos_vec.push_back(pos_4);
        pos_vec.push_back(pos_7);
        pos_vec.push_back(pos_3);
        pos_vec.push_back(pos_0);

        pos_vec.push_back(pos_1);
        pos_vec.push_back(pos_2);
        pos_vec.push_back(pos_6);
        pos_vec.push_back(pos_5);
    }

    return pos_vec;
}

void initFunction()
{
    cout << "loading mesh ......" << endl;
    mesh = static_cast<CubicMesh<double> *>(VolumetricMeshIO<double, 3>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Vol_Mesh/tire.smesh"));
    cout << "load mesh successfully" << endl;

    //---------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    //---------------------------------------------------------------------------------------------------------------------
    std::vector<Vector3d>  pos_vec = getVolumetricMeshQuads(mesh);

    auto quad_render_util = make_shared<QuadRenderUtil>();
    quad_render_util->setQuads(pos_vec);

    //---------------------------------------------------------------------------------------------------------------------
    auto quad_wireframe_render_task = make_shared<QuadWireframeRenderTask>(quad_render_util);
    render_scene_config.pushBackRenderTask(quad_wireframe_render_task);

    //---------------------------------------------------------------------------------------------------------------------
    auto quad_solid_render_task = make_shared<QuadSolidRenderTask>(quad_render_util);
    
    ColorMap<float> color_map;
    vector<Color4f> quad_col_vec;
    for (int i = 0; i < pos_vec.size() / 4; ++i)
        quad_col_vec.push_back(color_map.colorViaRatio(float(i) / (pos_vec.size() / 4)));

    //quad_solid_render_task->setQuadColors(quad_col_vec);
    //render_scene_config.pushBackRenderTask(quad_solid_render_task);

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

    //auto flash_light = make_shared<FlashLight>();
    //flash_light->setAmbient(Color4f::White());
    //render_scene_config.pushBackLight(flash_light);

    
    auto flash_light_2 = make_shared<FlashLight>();
    flash_light_2->setAmbient(Color4f::White());
    render_scene_config.pushBackLight(flash_light_2);

    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

    glClearDepth(1.0);
    glClearColor(0.49, 0.49, 0.49, 1.0);

    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
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