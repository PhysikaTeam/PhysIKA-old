/*
* @file fluid_render_task_test.cpp
* @brief Test for fluid render task.
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
#include <fstream>
#include <string>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Vectors/vector_4d.h"
#include "Physika_Core/Utilities/physika_assert.h"

#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/Lights/directional_light.h"
#include "Physika_Render/Lights/point_light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flash_light.h"


#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Plane_Render/plane_render_task.h"

#include "Physika_Render/Color/color.h"

#include "Physika_Render/Screen_Based_Render_Manager/screen_based_render_manager.h"
#include "Physika_Render/Fluid_Render/fluid_render_util.h"
#include "Physika_Render/Fluid_Render/fluid_point_render_task.h"
#include "Physika_Render/Fluid_Render/fluid_render_task.h"

using namespace std;
using namespace Physika;


GLfloat * readVec4File(const string & file_name, int & num)
{
    fstream file(file_name, ios::in);
    if (file.good() == false)
    {
        cerr << "can't open file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    file >> num;
    cout << "file_name: " << file_name << endl;
    cout << "num: " << num << endl;

    GLfloat * data = new GLfloat[4 * num];
    for (int i = 0; i < num; ++i)
        file >> data[4 * i] >> data[4 * i + 1] >> data[4 * i + 2] >> data[4 * i + 3];

    return data;
}

GLfloat * readDensityFile(const string & file_name, int & num)
{
    fstream file(file_name, ios::in);
    if (file.good() == false)
    {
        cerr << "can't open file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    file >> num;
    cout << "file_name: " << file_name << endl;
    cout << "num: " << num << endl;

    GLfloat * data = new GLfloat[num];
    for (int i = 0; i < num; ++i)
        file >> data[i];

    return data;
}

GLuint * readIndicesFile(const string & file_name, int & num)
{
    fstream file(file_name, ios::in);
    if (file.good() == false)
    {
        cerr << "can't open file: " << file_name << endl;
        exit(EXIT_FAILURE);
    }

    file >> num;
    cout << "file_name: " << file_name << endl;
    cout << "num: " << num << endl;

    GLuint * data = new GLuint[num];
    for (int i = 0; i < num; ++i)
        file >> data[i];

    return data;
}

void initFunction()
{

    //----------------------------------------------------------------------------------------------------------------------------------------------------------------
    int fluid_particle_num;
    int id = 0;

    cout << "read data ..." << endl;

    string path = "Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Screen_Based_Render_Manager/";

    GLfloat * pos = readVec4File(path + "data/pos_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_1 = readVec4File(path + "data/anisotropy_1_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_2 = readVec4File(path + "data/anisotropy_2_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_3 = readVec4File(path + "data/anisotropy_3_" + to_string(id) + ".txt", fluid_particle_num);

    GLfloat * density = readDensityFile(path + "data/density_" + to_string(id) + ".txt", fluid_particle_num);
    GLuint  * indices = readIndicesFile(path + "data/indice_" + to_string(id) + ".txt", fluid_particle_num);

    cout << "fluid particle num: " << fluid_particle_num << endl;

    cout << "read data successfully!" << endl;

    //-----------------------------------------------------------------------------------------------------------------------------------------
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();

    //-----------------------------------------------------------------------------------------------------------------------------------------

    auto fluid_render_util = make_shared<FluidRenderUtil>(fluid_particle_num, 0);
    fluid_render_util->updateFluidParticleBuffer(pos, density, anisotropy_1, anisotropy_2, anisotropy_3, indices, fluid_particle_num);

    auto fluid_point_render_task = make_shared<FluidPointRenderTask>(fluid_render_util);
    //render_scene_config.pushBackRenderTask(fluid_point_render_task);

    auto fluid_render_task = make_shared<FluidRenderTask>(fluid_render_util);
    render_scene_config.pushBackRenderTask(fluid_render_task);

    //-----------------------------------------------------------------------------------------------------------------------------------------
    auto plane_render_task = make_shared<PlaneRenderTask>();
    plane_render_task->addPlane({ 0.0f, 1.0f, 0.0f, 0.0f });
    render_scene_config.pushBackRenderTask(plane_render_task);

    //-----------------------------------------------------------------------------------------------------------------------------------------
    //light config

    auto spot_light = make_shared<SpotLight>();
    spot_light->setPos({ 0, 50, 0 });
    spot_light->setSpotDirection({ 0, -1, 0 });
    spot_light->setSpotCutoff(0.0);
    spot_light->setSpotOuterCutoff(20);
    spot_light->setAmbient(Color4f::Gray());
    render_scene_config.pushBackLight(spot_light);

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
    GlutWindow glut_window("screen_based_fluid_test", 1280, 720);

    cout << "Window name: " << glut_window.name() << "\n";
    cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    render_scene_config.setCameraPosition(Vector<double, 3>(0, 0, 200));
    render_scene_config.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    render_scene_config.setCameraNearClip(0.001);
    render_scene_config.setCameraFarClip(1.0e3);

    glut_window.setDisplayFunction(displayFunction);
    glut_window.setInitFunction(initFunction);
    glut_window.createWindow();
    return 0;
}