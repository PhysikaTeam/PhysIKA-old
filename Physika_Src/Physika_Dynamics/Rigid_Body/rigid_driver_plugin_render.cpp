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
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim>
RigidDriverPluginRender<Scalar, Dim>::RigidDriverPluginRender():
	window_(NULL)
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
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onRun()
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onAdvanceFrame()
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onInitialize()
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onAdvanceStep(Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body)
{
	RenderBase* render;
	switch(rigid_body->objectType())
	{
	case CollidableObject<Scalar, Dim>::MESH_BASED: render = new SurfaceMeshRender<Scalar>();;break;
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
void RigidDriverPluginRender<Scalar, Dim>::setDriver(RigidBodyDriver<Scalar, Dim>* driver)
{
	this->driver_ = driver;
	if(driver == NULL || window_ == NULL)
		return;
	unsigned int num_rigid_body = this->driver_->numRigidBody();
	for(unsigned int i = 0; i < num_rigid_body; ++i)
	{
		onAddRigidBody(this->driver_->rigidBody(i));
	}
}

template <typename Scalar,int Dim>
void RigidDriverPluginRender<Scalar, Dim>::setWindow(GlutWindow* window)
{
	window_ = window;
	//window_->setIdleFunction(&RigidDriverPluginRender<Scalar, Dim>::idle);
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

}

template <typename Scalar,int Dim>
RigidDriverPluginRender<Scalar, Dim>* RigidDriverPluginRender<Scalar, Dim>::active_render_;

//explicit instantiation
template class RigidDriverPluginRender<float, 3>;
template class RigidDriverPluginRender<double, 3>;

}