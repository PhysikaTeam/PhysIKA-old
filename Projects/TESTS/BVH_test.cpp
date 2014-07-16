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
#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh_node.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume_kdop18.h"
#include "Physika_Geometry/Bounding_Volume/bounding_volume.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Utilities/Timer/timer.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"

#include <GL/freeglut.h>
#include <GL/glui.h>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"

#include "Physika_Dynamics/Rigid_Body/rigid_body.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin_render.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin_print.h"

using namespace std;
using namespace Physika;

Transform<double> *trans;
unsigned int render_i = 0;

int main()
{
    SurfaceMesh<double> mesh_ball;
    if(!ObjMeshIO<double>::load(string("ball_high.obj"), &mesh_ball))
		exit(1);

    SurfaceMesh<double> mesh_box;

    if(!ObjMeshIO<double>::load(string("box_tri.obj"), &mesh_box))
        exit(1);

    cout<<mesh_box.numFaces();
    for(unsigned int i = 0; i <mesh_box.numFaces(); ++i)
    {
        Face<double> *face = mesh_box.facePtr(i);
        face->printVertices();
        if(i == 12)
            cout<<face->numVertices()<<"\n";
    }

    mesh_ball.computeAllFaceNormals();
    mesh_box.computeAllFaceNormals();
    
	RigidBodyDriver<double, 3> driver;
    driver.setGravity(0.981);

	RigidBody<double,3> ball(&mesh_ball);
    ball.setCoeffFriction(0.5);

    RigidBody<double,3> box(&mesh_box);
    //box.setCoeffFriction(0.5);
    //box.setCoeffRestitution(1);

    RigidBody<double,3> floor(&mesh_box);
    floor.setTranslation(Vector<double, 3>(0, -5, 0));
    floor.setScale(Vector<double, 3>(100, 1, 100));
    //floor.setCoeffFriction(0.5);
    floor.setFixed(true);

    RigidBody<double,3>* object;

    for(int i = 0; i < 3; i++)
    {
        for(int j = 1; j <= 3; j++)
        {
            for(int k = 0; k < 3; k++)
            {
                object = new RigidBody<double,3>(box);
                object->setTranslation(Vector<double, 3>(i*2-2, j*2, k*2));
                object->setRotation(Vector<double, 3>(0.5, 0.5, 0.5));
                driver.addRigidBody(object);
            }
        }
    }

    driver.addRigidBody(&floor);




    GlutWindow glut_window;
    cout<<"Window name: "<<glut_window.name()<<"\n";
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    //glut_window.setCameraPosition(Vector<double, 3>(20, 0, 0));
	glut_window.setCameraPosition(Vector<double, 3>(0, 0, 20));
	glut_window.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
    //glut_window.setDisplayFunction(display);
	glut_window.setCameraFarClip(10000);
	glut_window.setCameraNearClip(1.0e-3);
	glut_window.setCameraFOV(45);
	
	RigidDriverPluginRender<double, 3>* plugin = new RigidDriverPluginRender<double, 3>();
	plugin->setWindow(&glut_window);
	driver.addPlugin(plugin);
	//plugin->disableRenderSolidAll();
	plugin->enableRenderWireframeAll();
    //plugin->enableRenderContactFaceAll();
    //plugin->enableRenderContactNormalAll();

    RigidDriverPluginPrint<double, 3>* print_plugin = new RigidDriverPluginPrint<double, 3>();
    driver.addPlugin(print_plugin);


    glut_window.createWindow();


    return 0;
}
