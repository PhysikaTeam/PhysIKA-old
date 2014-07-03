/*
 * @file glut_window_test.cpp 
 * @brief Test VolumetricMeshRender of Physika.
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

#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render.h"
#include "Physika_Core/Transform/transform.h"

using namespace std;
using namespace Physika;

// tri
double a[6]={0,0,0,1,1,0};
unsigned int element[3]={0,1,2};
int b = 3;
TriMesh<double> tri_mesh(3,a,1,element);

// cubic
double cubicvp[24]={0,0,0 ,1,0,0, 1,1,0 ,0,1,0 ,0,0,1 ,1,0,1, 1,1,1 ,0,1,1};
unsigned int elecubic[8]={0,1,2,3,4,5,6,7};
CubicMesh<double> cubic_mesh(8,cubicvp,1,elecubic);


// quad
double quadvp[8]={0,0,0,1,1,0,1,1};
unsigned int elequad[4]={0,2,3,1};
QuadMesh<double> quad_mesh(4,quadvp,1,elequad);

// tet
double tetvp[21]={0,0,0, 0,0,1, 0,1,0, 1,0,0, 0,0,2, 0,2,0, 2,0,0};
unsigned int eletet[8]={0,1,2,3,0,4,5,6};
TetMesh<double> tet_mesh(7,tetvp,2,eletet);

VolumetricMeshRender<double,3> meshRender;
VolumetricMeshRender<double,3> meshRender2;
VolumetricMesh<double,3>* vol_mesh;

vector<unsigned int> element_id;
vector<unsigned int> vertex_id;
Color<float> color(1.0,1.0,0.0);
vector<Color<float>> color_vector;

void displayFunction()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
    //cur_window->orbitCameraRight(0.1);
    (cur_window->camera()).look();

    meshRender.disableRenderSolid();
	//meshRender.disableRenderWireframe();
	//meshRender.disableRenderVertices();
	//meshRender.enableRenderVertices();
	meshRender.enableRenderWireframe();
	glColor4f(1,0,0,1.0);
	meshRender.renderSolidWithAlpha(0.05);
	glColor4f(0,1,0,1.0);
	meshRender.render();
	//meshRender.renderVertexWithColor(vertex_id,color);
	//meshRender.renderElementWithColor(element_id,color);
	//glTranslatef(3,0,0);
	//meshRender2.renderSolidWithAlpha(0.4);

    cur_window->displayFrameRate();
    glutSwapBuffers();
}

void idleFunction()
{
    cout<<"Custom idle function\n";
}

void initFunction()
{
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClearDepth(1.0);

    for(unsigned int i=0; i<50; i++)
		for(unsigned int j=0; j<40; j++)
	{
		color_vector.push_back(Color<float>(0.2*i,0.25*j,0.1*(i+j)));
	}
	
	//color_vector.push_back(Color<float>(1,0,0));
	//color_vector.push_back(Color<float>(0,1,0));
	vol_mesh = VolumetricMeshIO<double,3>::load("e.smesh");

	meshRender.setVolumetricMesh(vol_mesh);
	vol_mesh->printInfo();
	cout<<"vertNum:"<<vol_mesh->vertNum()<<endl;
	cout<<"eleNum:"<<vol_mesh->eleNum()<<endl;
	cout<<"regionNum:"<<vol_mesh->regionNum()<<endl;

	meshRender2.setVolumetricMesh(&cubic_mesh);
	glColor4f(1,0,0,1.0);
	glPointSize(2.0);

	for(unsigned i=0; i<200;i++)
	{
		element_id.push_back(i);
		vertex_id.push_back(i);
	}
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