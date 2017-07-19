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
#include <fstream>
#include <string>
#include <GL/glew.h>
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
#include "Physika_Render/Screen_Based_Render_Manager/Fluid_Render/fluid_render.h"

using namespace std;
using namespace Physika;


GlutWindow * glut_window_ptr = NULL;
ScreenBasedRenderManager * screen_based_render_manager_ptr = NULL;
FluidRender * fluid_render_ptr = NULL;

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
        file >>data[i];

    return data;
}

void initFunction()
{

    screen_based_render_manager_ptr = new ScreenBasedRenderManager(glut_window_ptr);
    
    screen_based_render_manager_ptr->addPlane(Vector< float, 4>( 0.0f, 1.0f,  0.0f, 0.0f));
    screen_based_render_manager_ptr->addPlane(Vector< float, 4>( 0.0f, 0.0f,  1.0f, 0.51625f));
    //screen_based_render_manager_ptr->addPlane(Vector< float, 4>( 0.0f, 0.0f, -1.0f, 0.991574f));
    screen_based_render_manager_ptr->addPlane(Vector< float, 4>( 1.0f, 0.0f,  0.0f, 0.0165746f));
    screen_based_render_manager_ptr->addPlane(Vector< float, 4>(-1.0f, 0.0f,  0.0f, 3.01625f));
    


    screen_based_render_manager_ptr->setLightPos({ 4.35653f, 12.5529f, 5.77261f });
    screen_based_render_manager_ptr->setLightTarget({ 1.508125f, 1.00766f, 0.0f });
    screen_based_render_manager_ptr->setLightFov(28.0725f);
    screen_based_render_manager_ptr->setLightSpotMin(0.0);
    screen_based_render_manager_ptr->setLightSpotMax(0.5);

    int fluid_particle_num;
    int id = 0;

    cout << "read data ..." << endl;

    string path = "H:/Physika/Physika_Test/Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Screen_Based_Render_Manager/";

    GLfloat * pos          = readVec4File(path + "data/pos_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_1 = readVec4File(path + "data/anisotropy_1_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_2 = readVec4File(path + "data/anisotropy_2_" + to_string(id) + ".txt", fluid_particle_num);
    GLfloat * anisotropy_3 = readVec4File(path + "data/anisotropy_3_" + to_string(id) + ".txt", fluid_particle_num);

    GLfloat * density = readDensityFile(path + "data/density_" + to_string(id) + ".txt", fluid_particle_num);
    GLuint  * indices = readIndicesFile(path + "data/indice_" + to_string(id) + ".txt", fluid_particle_num);

    cout << "fluid particle num: " << fluid_particle_num << endl;

    cout << "read data successfully!" << endl;

    fluid_render_ptr = new FluidRender(fluid_particle_num, 0, 1280, 720);
    fluid_render_ptr->updateFluidParticleBuffer(pos, density, anisotropy_1, anisotropy_2, anisotropy_3, indices, fluid_particle_num);

    fluid_render_ptr->setDrawFluidParticle(false);
    fluid_render_ptr->setDrawPoint(true);

    screen_based_render_manager_ptr->setFluidRender(fluid_render_ptr);

    {
        std::string e = reinterpret_cast<const char *>(glGetString(GL_EXTENSIONS));
        std::cout << (e.find("GL_EXT_geometry_shader4") == std::string::npos ? "not found" : "GL_EXT_geometry_shader4") << std::endl;

        GLint major, minor;

        glGetIntegerv(GL_MAJOR_VERSION, &major);
        glGetIntegerv(GL_MINOR_VERSION, &minor);
        std::cout << "version : " << major << ", " << minor << std::endl;
        std::cout << std::boolalpha << (glutExtensionSupported("GL_EXT_geometry_shader4") == 0) << std::endl;
    }
}

void displayFunction()
{
    screen_based_render_manager_ptr->render();
}

int main()
{
    GlutWindow glut_window("screen_based_fluid_render_test", 1280, 720);
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