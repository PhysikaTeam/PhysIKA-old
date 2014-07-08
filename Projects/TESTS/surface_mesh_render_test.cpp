/*
 * @file glut_window_test.cpp 
 * @brief Test SurfaceMeshRender of Physika.
 * @author WeiChen
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
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_Render/Color/color.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_IO/Image_IO/image_io.h"

using namespace std;
using namespace Physika;

SurfaceMesh<double> mesh;
SurfaceMeshRender<double> meshRender;
vector<unsigned int> face_id;
vector<unsigned int> vertex_id;
Color<float> color(1.0,0.0,0.0);
vector<Color<float>> color_vector;

void displayFunction()
{
    GLfloat light_position[4]={0.0,0.0,20.0,0.0};
	glLightfv(GL_LIGHT0,GL_POSITION,light_position);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
    //cur_window->orbitCameraRight(0.1);
    (cur_window->camera()).look();

    
    /***************************************************************/
    //meshRender.disableRenderSolid();
	//meshRender.disableRenderWireframe();
	//meshRender.disableRenderVertices();
	//meshRender.enableRenderVertices();
	//meshRender.enableRenderWireframe();
	//meshRender.disableTexture();
	//meshRender.enableTexture();
	meshRender.render();
    //meshRender.renderVertexWithColor(vertex_id,color);
	//meshRender.renderVertexWithColor(vertex_id,color_vector);
    //meshRender.renderFaceWithColor(face_id,color_vector);
	//meshRender.renderSolidWithCustomColor(color_vector);

    /*****************************************************************/

    cur_window->displayFrameRate();
    glutSwapBuffers();
}

void idleFunction()
{
    cout<<"Custom idle function\n";
}

void initFunction()
{
    ObjMeshIO<double>::load("fish.obj",&mesh);
	for(unsigned int i=0; i<50; i++)
	for(unsigned int j=0; j<40; j++)
	{
		color_vector.push_back(Color<float>(0.02*i,0.025*j,0.01*(i+j)));
	}
	meshRender.setSurfaceMesh(&mesh);
	meshRender.printInfo();
	getchar();
    	for(unsigned i=0; i<20000;i++)
	{
		face_id.push_back(i);
		vertex_id.push_back(i);
	}
    
    glClearDepth(1.0);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);	
	glClearColor( 0.49, 0.49, 0.49, 1.0 );

	glShadeModel( GL_SMOOTH );
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
    //cout<<"³õÊ¼»¯"<<endl;
    //getchar();
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
    //glut_window.setIdleFunction(idleFunction);
    glut_window.setInitFunction(initFunction);
    cout<<"Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    cout<<"Test window with GLUI controls:\n";

    // GLUI PAET
    /*
    GluiWindow glui_window;
    glui_window.setDisplayFunction(displayFunction);
    glui_window.setCameraPosition(Vector<double,3>(0,0,200));
    glui_window.setCameraFocusPosition(Vector<double,3>(0,0,0));
    glui_window.setCameraNearClip(0.001);
    glui_window.setCameraFarClip(1.0e4);
    glui_window.setBackgroundColor(Color<double>::White());
    glui_window.setTextColor(Color<double>::Black());
    glui_window.setKeyboardFunction(keyboardFunction);
    GLUI *glui = glui_window.gluiWindow();
    PHYSIKA_ASSERT(glui);
    glui->add_statictext("Simple Window with GLUI controls");
    glui_window.createWindow();
    */
    return 0;
}