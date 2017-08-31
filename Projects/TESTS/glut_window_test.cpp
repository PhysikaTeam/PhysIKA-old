/*
 * @file glut_window_test.cpp 
 * @brief Test GlutWindow and GluiWindow of Physika.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <string>
#include <iostream>
#include <GL/freeglut.h>
#include <GL/glui.h>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_IO/Surface_Mesh_IO/surface_mesh_io.h"
#include "Physika_Render/Lights/light.h"
using namespace std;
using Physika::GlutWindow;
using Physika::GluiWindow;
using Physika::Vector;
using Physika::Color;
using Physika::Light;
using Physika::SurfaceMesh;
using Physika::SurfaceMeshIO;
using Physika::SurfaceMeshRender;

void displayFunction()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
    //cur_window->orbitCameraRight(0.1);
    cur_window->applyCameraAndLights();
    
	//glLoadIdentity();
	//glTranslatef(0.0f,0.0f,-3.0f);			
	/*
	 * Triangle code starts here
	 * 3 verteces, 3 colors.
	 */
	// glBegin(GL_TRIANGLES);					
	// 	glColor3f(0.0f,0.0f,1.0f);			
	// 	glVertex3f( 0.0f, 1.0f, 0.0f);		
	// 	glColor3f(0.0f,1.0f,0.0f);			
	// 	glVertex3f(-1.0f,-1.0f, 0.0f);		
	// 	glColor3f(1.0f,0.0f,0.0f);			
	// 	glVertex3f( 1.0f,-1.0f, 0.0f);		
	// glEnd();	
    /* 
     * Test openGL primitive wrappers
     */
    openGLColor3(Color<float>::Blue());
    glutWireCube(100.0);
    openGLColor3(Color<float>::Red());
    glutSolidSphere(2, 30, 30);
    cur_window->displayFrameRate();
    glutSwapBuffers();
}

void idleFunction()
{
    glutSetWindow(GluiWindow::mainWindowId());
    GlutWindow *cur_window = static_cast<GlutWindow*>(glutGetWindowData());
    Light *default_light = cur_window->lightAtIndex(0);
    PHYSIKA_ASSERT(default_light);
    Vector<float,3> light_pos = default_light->position();
    light_pos[1] -= 5;
    default_light->setPosition(light_pos);
    glutPostRedisplay();
}

void initFunction()
{
    glClearColor(1.0, 0.0, 1.0, 1.0);
    glClearDepth(1.0);
}

void keyboardFunction(unsigned char key, int x, int y )
{
    GlutWindow *cur_window = static_cast<GlutWindow*>(glutGetWindowData());
    Light *default_light = cur_window->lightAtIndex(0);
    PHYSIKA_ASSERT(default_light);
    GlutWindow::bindDefaultKeys(key,x,y);
    switch(key)
    {
    case 't':
        cout<<"test\n";
        break;
    case 'l':
        default_light->turnOff();
        break;
    case 'o':
        default_light->turnOn();
        break;
    default:
        break;
    }
}

int main()
{
    // GlutWindow glut_window;
    // cout<<"Window name: "<<glut_window.name()<<"\n";
    // cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    // glut_window.setCameraPosition(Vector<double,3>(0,0,200));
    // glut_window.setCameraFocusPosition(Vector<double,3>(0,0,0));
    // glut_window.setCameraNearClip(0.001);
    // glut_window.setCameraFarClip(1.0e4);
    // glut_window.setDisplayFunction(displayFunction);
    // cout<<"Test GlutWindow with custom display function:\n";
    // glut_window.createWindow();
    cout<<"Test window with GLUI controls:\n";
    GluiWindow glui_window;
    glui_window.setCameraPosition(Vector<double,3>(0,0,200));
    glui_window.setCameraFocusPosition(Vector<double,3>(0,0,0));
    glui_window.setCameraNearClip(0.001);
    glui_window.setCameraFarClip(1.0e4);
    //glui_window.setBackgroundColor(Color<double>::White());
    //glui_window.setTextColor(Color<double>::Black());
    glui_window.setKeyboardFunction(keyboardFunction);
    glui_window.setIdleFunction(idleFunction);
    SurfaceMesh<double> mesh;
    string mesh_file_name("ball_high.obj");
    bool status = SurfaceMeshIO<double>::load(mesh_file_name,&mesh);
    SurfaceMeshRender<double> mesh_render(&mesh);
    glui_window.pushBackRenderTask(&mesh_render);
    GLUI *glui = glui_window.gluiWindow();
    PHYSIKA_ASSERT(glui);
    glui->add_statictext("Simple Window with GLUI controls");
    glui_window.createWindow();
    return 0;
}
