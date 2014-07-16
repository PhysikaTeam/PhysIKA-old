/*
 * @file rigid_driver_plugin_render.cpp
 * @Render plugin of rigid body driver.
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */


#include "Physika_Dynamics/Rigid_Body/rigid_body.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin_render.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Dynamics/Collidable_Objects/collision_pair.h"
#include "Physika_Dynamics/Collidable_Objects/contact_point.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include <GL/freeglut.h>

namespace Physika{

template <typename Scalar,int Dim>
RigidDriverPluginRender<Scalar, Dim>::RigidDriverPluginRender():
	window_(NULL),
	is_render_contact_face_(false),
    is_render_contact_normal_(false),
    contact_face_ids_(NULL),
    normal_length_(2)
{
	active();
}

template <typename Scalar,int Dim>
RigidDriverPluginRender<Scalar, Dim>::~RigidDriverPluginRender()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	for(unsigned int i = 0; i < num_render; ++i)
	{
		delete render_queue_[i];
	}
	render_queue_.clear();
    if(contact_face_ids_ != NULL)
        delete [] contact_face_ids_;
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onInitialize(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onBeginFrame(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onEndFrame(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onBeginTimeStep(Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onEndTimeStep(Scalar time, Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onWrite(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onRead(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onRestart(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onBeginRigidStep(unsigned int step, Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onEndRigidStep(unsigned int step, Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body)
{
	if(rigid_body == NULL)
		return;
	RenderBase* render;
	switch(rigid_body->objectType())
	{
	case CollidableObjectInternal::MESH_BASED: render = new SurfaceMeshRender<Scalar>();;break;
	default: std::cerr<<"Object type error!"<<std::endl; return;
	}
	SurfaceMeshRender<Scalar>* mesh_render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render);
	mesh_render->setSurfaceMesh(rigid_body->mesh());
	mesh_render->setTransform(rigid_body->transformPtr());
	render_queue_.push_back(render);
	if(window_ != NULL)
		window_->pushBackRenderTask(render);
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onBeginCollisionDetection()
{
	
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onEndCollisionDetection()
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::setDriver(DriverBase<Scalar>* driver)
{
	this->rigid_driver_ = dynamic_cast<RigidBodyDriver<Scalar, Dim>*>(driver);
	if(this->rigid_driver_ != NULL)
		this->driver_ = driver;
	else
		return;
	if(this->driver_ == NULL || window_ == NULL)
		return;
	unsigned int num_rigid_body = this->rigid_driver_->numRigidBody();
	for(unsigned int i = 0; i < num_rigid_body; ++i)
	{
		onAddRigidBody(this->rigid_driver_->rigidBody(i));
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::setWindow(GlutWindow* window)
{
	if(window == NULL)
		return;
	window_ = window;
	window_->setIdleFunction(&RigidDriverPluginRender<Scalar, Dim>::idle);
    window_->setDisplayFunction(&RigidDriverPluginRender<Scalar, Dim>::display);
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	for(unsigned int i = 0; i < num_render; ++i)
	{
		window_->pushBackRenderTask(render_queue_[i]);
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::active()
{
	RigidDriverPluginRender<Scalar, Dim>::active_render_ = this;
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::idle()
{
	active_render_->rigid_driver_->advanceStep(active_render_->rigid_driver_->computeTimeStep());

	if(active_render_->is_render_contact_face_)//get contact faces' ids
	{
        unsigned int num_body = active_render_->rigid_driver_->numRigidBody();
        if(active_render_->contact_face_ids_ != NULL)
        {
            delete [] active_render_->contact_face_ids_;
            active_render_->contact_face_ids_ = NULL;
        }
        active_render_->contact_face_ids_ = new std::vector<unsigned int>[num_body];

		CollisionDetectionResult<Scalar, Dim>* collision_result = &(active_render_->rigid_driver_->collisionResult());
		unsigned int num_collision = collision_result->numberCollision();
		for(unsigned int i = 0; i < num_collision; ++i)
        {
            CollisionPairBase<Scalar, Dim>* pair = collision_result->collisionPairs()[i];
            (active_render_->contact_face_ids_)[pair->objectLhsIdx()].push_back(pair->faceLhsIdx());
            (active_render_->contact_face_ids_)[pair->objectRhsIdx()].push_back(pair->faceRhsIdx());
        }
	}

    if(active_render_->is_render_contact_normal_)//get contact faces' normals
    {
        active_render_->contact_normal_positions_.clear();
        active_render_->contact_normal_orientation_.clear();
        ContactPointManager<Scalar, Dim>* contact_points = &(active_render_->rigid_driver_->contactPoints());
        unsigned int num_contact = contact_points->numContactPoint();
        for(unsigned int i = 0; i < num_contact; ++i)
        {
            active_render_->contact_normal_positions_.push_back(contact_points->contactPoint(i)->globalContactPosition());
            active_render_->contact_normal_orientation_.push_back(contact_points->contactPoint(i)->globalContactNormalLhs());
            active_render_->contact_normal_positions_.push_back(contact_points->contactPoint(i)->globalContactPosition());
            active_render_->contact_normal_orientation_.push_back(contact_points->contactPoint(i)->globalContactNormalRhs());
        }
    }

    glutPostRedisplay();

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::display()
{
    GlutWindow *window = active_render_->window_;
    PHYSIKA_ASSERT(window);
    Color<double> background_color = window->backgroundColor<double>();
    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    (window->camera()).look();  //set camera
    
    if(active_render_->is_render_contact_face_ && active_render_->contact_face_ids_ != NULL)//render contact faces
    {
        SurfaceMeshRender<Scalar>* render;
        unsigned int num_body = active_render_->rigid_driver_->numRigidBody();
        for(unsigned int i = 0; i < num_body; ++i)
        {
            render = dynamic_cast<SurfaceMeshRender<Scalar>*>(active_render_->render_queue_[i]);
            if(render == NULL)
                continue;
            //render->synchronize();
            render->renderFaceWithColor((active_render_->contact_face_ids_)[i], Color<float>::Blue());
        }
    }

    if(active_render_->is_render_contact_normal_)//render contact normals
    {
        glPushAttrib(GL_ALL_ATTRIB_BITS);
        glPushMatrix();
        openGLColor3(Color<double>::Red());
        glBegin(GL_LINES);
        unsigned int num_normal = static_cast<unsigned int>(active_render_->contact_normal_positions_.size());
        for(unsigned int i = 0; i < num_normal; ++i)
        {
            openGLVertex(active_render_->contact_normal_positions_[i]);
            openGLVertex(active_render_->contact_normal_positions_[i] + active_render_->contact_normal_orientation_[i] * active_render_->normal_length_);
        }
        glEnd();
        glPopMatrix();
        glPopAttrib();
    }

    (window->renderManager()).renderAll(); //render all tasks of render manager
    window->displayFrameRate();
    glutSwapBuffers();
}

template <typename Scalar,int Dim>
unsigned int RigidDriverPluginRender<Scalar, Dim>::numRender() const
{
	return static_cast<unsigned int>(render_queue_.size());
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderSolidAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->enableRenderSolid();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderSolidAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->disableRenderSolid();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderVerticesAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->enableRenderVertices();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderVerticesAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->disableRenderVertices();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderWireframeAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->enableRenderWireframe();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderWireframeAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->disableRenderWireframe();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableFlatShadingAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->enableFlatShading();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableSmoothShadingAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->enableSmoothShading();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableTextureAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->enableTexture();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableTextureAll()
{
	unsigned int num_render = static_cast<unsigned int>(render_queue_.size());
	SurfaceMeshRender<Scalar>* render;
	for(unsigned int i = 0; i < num_render; ++i)
	{
		render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[i]);
		if(render == NULL)
			continue;
		render->disableTexture();
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderSolidAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->enableRenderSolid();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderSolidAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->disableRenderSolid();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderVerticesAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->enableRenderVertices();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderVerticesAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->disableRenderVertices();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderWireframeAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->enableRenderWireframe();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderWireframeAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->disableRenderWireframe();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableFlatShadingAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->enableFlatShading();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableSmoothShadingAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->enableSmoothShading();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableTextureAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->enableTexture();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableTextureAt(unsigned int index)
{
	if(index >= numRender())
	{
		std::cerr<<"Render index out of range!"<<std::endl;
		return;
	}
	SurfaceMeshRender<Scalar>* render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_queue_[index]);
	if(render != NULL)
		render->disableTexture();
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderContactFaceAll()
{
	is_render_contact_face_ = true;
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderContactFaceAll()
{
    is_render_contact_face_ = false;
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::enableRenderContactNormalAll()
{
    is_render_contact_normal_ = true;
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::disableRenderContactNormalAll()
{
    is_render_contact_normal_ = false;
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::setNormalLength(Scalar normal_lenth)
{
    normal_length_ = normal_lenth;
}

template <typename Scalar,int Dim>
RigidDriverPluginRender<Scalar, Dim>* RigidDriverPluginRender<Scalar, Dim>::active_render_;

//explicit instantiation
template class RigidDriverPluginRender<float, 3>;
template class RigidDriverPluginRender<double, 3>;

}