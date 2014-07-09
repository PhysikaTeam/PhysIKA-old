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
#include "Physika_Render/Lights/light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/light_manager.h"
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
	cout<<GL_LIGHT0<<endl;
	cout<<GL_LIGHT1<<endl;
	cout<<GL_LIGHT2<<endl;
	cout<<GL_LIGHT3<<endl;
	cout<<GL_LIGHT4<<endl;
	cout<<GL_LIGHT5<<endl;
	cout<<GL_LIGHT6<<endl;
	cout<<GL_LIGHT7<<endl;
	cout<<"*******************************************************************************"<<endl;
	Light::printOccupyInfo();
    Light light1;
	Light::printOccupyInfo();
    Light light2(GL_LIGHT1);
	Light::printOccupyInfo();
    light1.setLightId(GL_LIGHT3);
	Light::printOccupyInfo();


	Light light3(GL_LIGHT1);
	Light::printOccupyInfo();
	Light light4(GL_LIGHT1);
	Light::printOccupyInfo();
	Light light5(GL_LIGHT1);
	Light::printOccupyInfo();
	Light light6(GL_LIGHT1);
	Light::printOccupyInfo();
	Light light7(GL_LIGHT1);
	Light::printOccupyInfo();
	//Light light8(GL_LIGHT1);
	//Light::printOccupyInfo();
	light7.~Light();
	Light::printOccupyInfo();
    light1.setAmbient(Color<float>(1.0,0.0,0.0,1.0));
    light1.setDiffuse(Color<float>(0.0,1.0,0.0,1.0));
    light1.setSpecular(Color<float>(0.0,0.0,1.0,1.0));
    light1.setConstantAttenuation(1.1);
    light1.setLinearAttenuation(1.2);
    light1.setQuadraticAttenuation(1.3);
    light1.setPosition(Vector<float,3>(2, 2, 2));
    light1.turnOn();
    cout<<light1;
	
	cout<<"**************************************************************************"<<endl;
    LightManager light_manager;
    light_manager.setLightModelAmbient(Color<double>(1.0,1.0,1.0,1.0));
    light_manager.setLightModelLocalViewer(true);
    light_manager.setLightModelTwoSide(false);
    //light_manager.setLightModelColorControl(GL_SEPARATE_SPECULAR_COLOR);
    light_manager.insertBack(&light1);
    light_manager.insertBack(&light2);
    light_manager.turnAllOn();
    light_manager.turnAllOff();
    //light_manager.insertFront(&light2);
    //light_manager.removeFront();
    //light_manager.insertBack(&light1);
    //light_manager.removeBack();
    //cout<<light_manager.lightAtIndex(0)->lightId()<<endl;
    //cout<<light_manager.lightIndex(&light2)<<endl;
    //light_manager.turnLightOnAtIndex(1);
    //light_manager.turnLightOffAtIndex(1);
    //light_manager.insertBack(&light1);
    //light_manager.insertBack(&light1);
    //light_manager.insertBack(&light1);
    //light_manager.insertBack(&light1);
    light_manager.insertAtIndex(0,&light1);
    //light_manager.insertBack(&light0);
    //light_manager.insertBack(&light0);
    light_manager.printInfo();
    getchar();
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
	//for(int i=0; i<8;i++)
		//cout<<Light::is_occupied_[i];
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
