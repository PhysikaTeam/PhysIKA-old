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

#include <GL/glut.h>
#include <GL/glui.h>
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_GUI/Glui_Window/glui_window.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"

#include "Physika_Dynamics/Rigid_Body/rigid_body.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin_render.h"

using namespace std;
using namespace Physika;

int main()
{
    SurfaceMesh<double> mesh_ball;
    if(!ObjMeshIO<double>::load(string("ball_high.obj"), &mesh_ball))
		exit(1);

	RigidBodyDriver<double, 3> driver;

	RigidBody<double,3> body1;
	body1.setMesh(&mesh_ball);

	RigidBody<double,3> body2;
	body2.setMesh(&mesh_ball);
	body2.transformPtr()->setTranslation(Vector<double, 3>(0, 55, 0));

	driver.addRigidBody(&body1);
	driver.addRigidBody(&body2);

    GlutWindow glut_window;
    cout<<"Window name: "<<glut_window.name()<<"\n";
    cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
	glut_window.setCameraPosition(Vector<double, 3>(-60, 0, 60));
	glut_window.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
	glut_window.setCameraFarClip(10000);
	glut_window.setCameraNearClip(1.0e-3);
	glut_window.setCameraFOV(45);
	
	RigidDriverPluginRender<double, 3>* plugin = new RigidDriverPluginRender<double, 3>();
	plugin->setWindow(&glut_window);
	driver.addPlugin(plugin);


    cout<<"Test GlutWindow with custom display function:\n";
    glut_window.createWindow();
    //glut_window.setIdleFunction(idleFunction);
    //cout<<"Window size: "<<glut_window.width()<<"x"<<glut_window.height()<<"\n";
    cout<<"Test window with GLUI controls:\n";




	system("pause");

    return 0;
}
