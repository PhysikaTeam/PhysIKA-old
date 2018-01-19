/*
 * @file surface_mesh_render_test.cpp 
 * @brief Test SurfaceMeshRender of Physika.
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
#include <GL/freeglut.h>
#include <GL/glui.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"


using namespace std;
using namespace Physika;

SurfaceMesh<double> mesh;
SurfaceMeshRender<double> meshRender;
vector<SurfaceMeshRender<double> > meshRender_vec;
vector<SurfaceMesh<double> > mesh_vec;


vector<unsigned int> face_id;
vector<unsigned int> vertex_id;
Color<float> color(1.0,0.0,0.0);
vector<Color<float>> color_vector;

void displayFunction()
{

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
    GlutWindow *cur_window = (GlutWindow*)glutGetWindowData();
	
    //(cur_window->camera()).look();
	//cur_window->applyCameraAndLights();

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


	/******************************************************************/
	// test for separateByGroup

	//for(int i=0; i<20; i++)
	//	meshRender_vec[i].render();
	/******************************************************************/

	//cout<<"light 0: "<<cur_window->lightAtIndex(0)->position()<<endl;
    cur_window->displayFrameRate();
    glutSwapBuffers();
}

void idleFunction()
{
    cout<<"Custom idle function\n";
}

void initFunction()
{
    ObjMeshIO<double>::load("Physika_Test_Src/Physika_Non_Unit_Test_Src/Physika_Render/Obj_Mesh/scene_dense_mesh_refine_texture.obj",&mesh);

	/***********************************************************************/
	for(unsigned int i=0; i<50; i++)
		for(unsigned int j=0; j<40; j++)
		{
			color_vector.push_back(Color<float>(0.02*i,0.025*j,0.01*(i+j)));
		}
	for(unsigned i=0; i<20000;i++)
	{
		face_id.push_back(i);
		vertex_id.push_back(i);
	}

	/***********************************************************************/
	// note: we have to set our light in initFunction, otherwise the setting will not be valid.

	//GlutWindow * cur_window = (GlutWindow*)glutGetWindowData();
	//cur_window->lightAtIndex(0)->setPosition(Vector<float,3>(0,50,0));
	//cur_window->lightAtIndex(0)->turnOn();

	meshRender.setSurfaceMesh(&mesh);
    //meshRender.enableDisplayList();
	meshRender.printInfo();

	/******************************************************************************/
	//test for separateByGroup

	//mesh.separateByGroup(mesh_vec);
	//cout<<"mesh_vec size: "<<mesh_vec.size()<<endl;
	//cout<<"mesh_vec[0] numMaterial: "<<mesh_vec[0].numMaterials()<<endl;
	//for(int i=0; i<mesh_vec.size(); i++)
	//{
	//	meshRender_vec.push_back(SurfaceMeshRender<double>());
	//	meshRender_vec[i].setSurfaceMesh(&mesh_vec[i]);
	//}
   /*******************************************************************************/
    
    glClearDepth(1.0);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);	
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);	
	glClearColor( 0.49, 0.49, 0.49, 1.0 );

	glShadeModel( GL_SMOOTH );
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_LIGHTING);
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
    
    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    render_scene_config.setCameraPosition(Vector<double,3>(0,0,200));
    render_scene_config.setCameraFocusPosition(Vector<double,3>(0,0,0));
    render_scene_config.setCameraNearClip(0.001);
    render_scene_config.setCameraFarClip(1.0e4);
    
    glut_window.setDisplayFunction(displayFunction);
    //glut_window.setIdleFunction(idleFunction);
    glut_window.setInitFunction(initFunction);
    cout<<"Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    cout<<"Test window with GLUI controls:\n";
	
    return 0;
}