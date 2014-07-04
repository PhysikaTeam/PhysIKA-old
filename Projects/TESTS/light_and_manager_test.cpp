/*
 * @file light_and_manager_test.cpp 
 * @brief Test Light and LightManager of Physika.
 * @author Wei Chen
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
#include <GL/freeglut.h>
#include <GL/glui.h>
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_GUI/Light/light.h"
#include "Physika_GUI/Light/spot_light.h"
#include "Physika_GUI/Light_Manager/light_manager.h"
using namespace std;
using namespace Physika;

void displayFunction()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
    //cur_window->orbitCameraRight(0.1);
    (cur_window->camera()).look();
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
    cout<<"Custom idle function\n";
}

void initFunction()
{
    glClearColor(1.0, 0.0, 1.0, 1.0);
    glClearDepth(1.0);

/***************************************************************/
    //LightManager light_manager;
    Light light0;
    Light light1(GL_LIGHT1);
    light0.setLightId(GL_LIGHT0);

    light0.setAmbient(Color<float>(1.0,0.0,0.0,1.0));
    light0.setDiffuse(Color<float>(0.0,1.0,0.0,1.0));
    light0.setSpecular(Color<float>(0.0,0.0,1.0,1.0));
    light0.setConstantAttenuation(1.1);
    light0.setLinearAttenuation(1.2);
    light0.setQuadraticAttenuation(1.3);
    light0.setPosition(Vector<float,3>(2.0,2.0,2.0));
    light0.turnOn();
    light0.turnOff();
    cout<<light0;

    LightManager light_manager;
    light_manager.setLightModelAmbient(Color<float>(1.0,1.0,1.0,1.0));
    light_manager.setLightModelLocalViewer(true);
    light_manager.setLightModelTwoSide(false);
    //light_manager.setLightModelColorControl(GL_SEPARATE_SPECULAR_COLOR);
    light_manager.insertBack(&light0);
    light_manager.insertBack(&light1);
    light_manager.turnAllOn();
    light_manager.turnAllOff();
    light_manager.insertFront(&light1);
    light_manager.removeFront();
    light_manager.insertBack(&light0);
    light_manager.removeBack();
    cout<<light_manager.lightAtIndex(0)->lightId()<<endl;
    cout<<light_manager.lightIndex(&light0)<<endl;
    light_manager.turnLightOnAtIndex(1);
    light_manager.turnLightOffAtIndex(1);
    light_manager.insertBack(&light0);
    light_manager.insertBack(&light0);
    light_manager.insertBack(&light0);
    light_manager.insertBack(&light0);
    light_manager.insertAtIndex(0,&light0);
   // light_manager.insertBack(&light0);
    //light_manager.insertBack(&light0);
    light_manager.printInfo();
    cout<<"num: "<<light_manager.numLights()<<endl;
    cout<<light_manager;
    system("pause");
}

void keyboardFunction(unsigned char key, int x, int y )
{
    GlutWindow::bindDefaultKeys(key,x,y);
    switch(key)
    {
    case 't':
        cout<<"test\n";
        break;
    default:
        break;
    }
}

int main()
{
    GlutWindow glut_window;
    cout<<"Window name: "<<glut_window.name()<<"\n";
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    glut_window.setCameraPosition(Vector<double,3>(0,0,200));
    glut_window.setCameraFocusPosition(Vector<double,3>(0,0,0));
    glut_window.setCameraNearClip(0.001);
    glut_window.setCameraFarClip(1.0e4);
    glut_window.setDisplayFunction(displayFunction);
    glut_window.setInitFunction(initFunction);
    cout<<"Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    glut_window.setIdleFunction(idleFunction);
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    cout<<"Test window with GLUI controls:\n";
    return 0;
}