/*
 * @file volumetric_mesh_test.cpp 
 * @brief Test the various types of volumetric meshes.
 * @author Mike Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include<iostream>
#include<string>
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/bvh_node_base.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh_node.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/Timer/timer.h"

#include <GL/glut.h>
#include <GL/glui.h>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"

using namespace std;
using namespace Physika;


MeshBasedCollidableObject<double, 3>* pObject1, *pObject2;

void displayFunction()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);		     // Clear Screen and Depth Buffer
	glLoadIdentity();
	gluLookAt(10, 0, -110, 0, 0, 0, 0, 1, 0);

	SurfaceMeshRender<double> meshRender1;
	meshRender1.setSurfaceMesh(pObject1->getMesh());
	//meshRender1.render();

	Vector<double, 3> pos = pObject2->transform().position();
	glTranslatef(pos[0], pos[1], pos[2]);	
	cout<<pos<<endl;
 
	SurfaceMeshRender<double> meshRender2;
	meshRender2.setSurfaceMesh(pObject2->getMesh());
	meshRender2.render();
	glTranslatef(-1*pos[0], -1*pos[1], -1*pos[2]);	

    glutSwapBuffers();
}

void initFunction()
{
	
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    glMatrixMode(GL_PROJECTION);												// select projection matrix
    glViewport(0, 0, width, height);        									// set the viewport
    glMatrixMode(GL_PROJECTION);												// set matrix mode
    glLoadIdentity();															// reset projection matrix
    gluPerspective(45.0,(GLdouble)width/height,1.0e-3,10000);           		// set up a perspective projection matrix
    glMatrixMode(GL_MODELVIEW);													// specify which matrix is the current matrix
    glShadeModel( GL_SMOOTH );
    glClearDepth( 1.0 );														// specify the clear value for the depth buffer
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );						// specify implementation-specific hints
    Color<unsigned char> black = Color<unsigned char>::Black();
    glClearColor(black.redChannel(), black.greenChannel(), black.blueChannel(), black.alphaChannel());	
}

int main()
{

	BoundingVolumeKDOP18<double, 3> KDOP;
	Vector<double, 3> point1(1, 1, 1);
	Vector<double, 3> point2(1, 1, -1);
	Vector<double, 3> point3(1, -1, 1);
	Vector<double, 3> point4(1, -1, -1);
	Vector<double, 3> point5(-1, 1, 1);
	Vector<double, 3> point6(-1, 1, -1);
	Vector<double, 3> point7(-1, -1, 1);
	Vector<double, 3> point8(-1, -1, -1);
	KDOP.unionWith(point1);
	KDOP.unionWith(point2);
	KDOP.unionWith(point3);
	KDOP.unionWith(point4);
	KDOP.unionWith(point5);
	KDOP.unionWith(point6);
	KDOP.unionWith(point7);
	KDOP.unionWith(point8);

	Vector<double, 3> point(-1, 1, 0.999);
	if(KDOP.isInside(point))
		cout<<"in"<<endl;

	Timer timer;
	
    SurfaceMesh<double> mesh_ball;
    if(!ObjMeshIO<double>::load(string("E:/Physika/ball_high.obj"), &mesh_ball))
		exit(1);
	
	pObject1 = new MeshBasedCollidableObject<double, 3>();
	pObject1->setMesh(&mesh_ball);
	ObjectBVH<double, 3> * pBVH1 = new ObjectBVH<double, 3>();
	pBVH1->setCollidableObject(pObject1);
	
	pObject2 = new MeshBasedCollidableObject<double, 3>();
	pObject2->setMesh(&mesh_ball);
	pObject2->transform().setPosition(Vector<double, 3>(10.15, 10.15, 10.15));
	ObjectBVH<double, 3> * pBVH2 = new ObjectBVH<double, 3>();
	pBVH2->setCollidableObject(pObject2);

	CollisionDetectionResult<double, 3> result;
	result.resetCollisionResults();

	timer.startTimer();
	if(pBVH1->collide(pBVH2, result))
		cout<<"collide"<<endl;
	timer.stopTimer();
	cout<<timer.getElapsedTime()<<endl;
	cout<<result.numberPCS()<<endl;
	cout<<result.numberCollision()<<endl;


    GlutWindow glut_window;
    cout<<"Window name: "<<glut_window.name()<<"\n";
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
	glut_window.setInitFunction(initFunction);
    glut_window.setDisplayFunction(displayFunction);
	
    cout<<"Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    //glut_window.setIdleFunction(idleFunction);
    //cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    cout<<"Test window with GLUI controls:\n";




	system("pause");

    return 0;
}
