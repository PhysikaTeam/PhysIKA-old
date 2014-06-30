/*
 * @file glut_window_test.cpp 
 * @brief Test GlutWindow and GluiWindow of Physika.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <GL/glut.h>
#include <GL/glui.h>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
using namespace std;
using Physika::GlutWindow;
using Physika::GluiWindow;
using Physika::Vector;

void displayFunction()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
	glLoadIdentity();
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
    //Physika::Color<unsigned char> color = Physika::Color<int>::convertColor<unsigned char>(Physika::Color<int>::Blue());
	glBegin(GL_TRIANGLES);					
    openGLColor3(Physika::Color<float>::Blue());
    //openGLColor3(color);
	openGLVertex(Vector<float,3>(0.0f, 1.0f, 0.0f));					
    openGLColor3(Physika::Color<float>::Green());
	openGLVertex(Vector<float,3>(-1.0f,-1.0f, 0.0f));					
    openGLColor3(Physika::Color<float>::Red());
	openGLVertex(Vector<float,3>(1.0f,-1.0f, 0.0f));
	glEnd();
    glutSwapBuffers();
}

void idleFunction()
{
    cout<<"Custom idle function\n";
}

void initFunction()
{
    glClearColor(1.0, 0.0, 1.0, 1.0);
    glClearDepth(1.0);
}

int main()
{
    GlutWindow glut_window;
    cout<<"Window name: "<<glut_window.name()<<"\n";
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    glut_window.setDisplayFunction(displayFunction);
    cout<<"Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    //glut_window.setIdleFunction(idleFunction);
    //cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    cout<<"Test window with GLUI controls:\n";
    GluiWindow glui_window;
    glui_window.setDisplayFunction(displayFunction);
    GLUI *glui = glui_window.gluiWindow();
    glui->add_statictext("Simple Window with GLUI controls");
    glui_window.createWindow();
    return 0;
}
