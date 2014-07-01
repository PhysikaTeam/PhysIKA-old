/*
 * @file rigid_body_driver.cpp
 * @Basic rigid body driver class.
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
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Render/Render_Base/render_base.h"
#include "Physika_Render/Surface_Mesh_Render/surface_mesh_render.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

namespace Physika{

template <typename Scalar,int Dim>
RigidBodyArchive<Scalar, Dim>::RigidBodyArchive():
	index_(0),
	rigid_body_(NULL),
	collide_object_(NULL),
	object_bvh_(NULL),
	render_(NULL)
{

}

template <typename Scalar,int Dim>
RigidBodyArchive<Scalar, Dim>::RigidBodyArchive(RigidBody<Scalar, Dim>* rigid_body)
{
	setRigidBody(rigid_body);
}

template <typename Scalar,int Dim>
RigidBodyArchive<Scalar, Dim>::~RigidBodyArchive()
{
	delete collide_object_;
	delete object_bvh_;
	delete render_;
}

template <typename Scalar,int Dim>
void RigidBodyArchive<Scalar, Dim>::setRigidBody(RigidBody<Scalar, Dim>* rigid_body)
{
	if(rigid_body == NULL)
		return;

	rigid_body_ = rigid_body;

	switch(rigid_body->objectType())
	{
		case CollidableObject<Scalar, Dim>::MESH_BASED: collide_object_ = new MeshBasedCollidableObject<Scalar, Dim>();break;
		default: std::cerr<<"Object type error!"<<std::endl; return;
	}
	MeshBasedCollidableObject<Scalar, Dim>* mesh_object = dynamic_cast<MeshBasedCollidableObject<Scalar, Dim>*>(collide_object_);
	mesh_object->setMesh(rigid_body->mesh());
	mesh_object->setTransform(rigid_body->transformPtr());

	object_bvh_ = new ObjectBVH<Scalar, Dim>();
	object_bvh_->setCollidableObject(collide_object_);

	render_ = new SurfaceMeshRender<Scalar>();
	SurfaceMeshRender<Scalar>* mesh_render = dynamic_cast<SurfaceMeshRender<Scalar>*>(render_);
	mesh_render->setSurfaceMesh(rigid_body->mesh());

}

template <typename Scalar,int Dim>
unsigned int RigidBodyArchive<Scalar, Dim>::index() const
{
	return index_;
}

template <typename Scalar,int Dim>
void RigidBodyArchive<Scalar, Dim>::setIndex(unsigned int index)
{
	index_ = index;
}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>* RigidBodyArchive<Scalar, Dim>::rigidBody()
{
	return rigid_body_;
}

template <typename Scalar,int Dim>
CollidableObject<Scalar, Dim>* RigidBodyArchive<Scalar, Dim>::collideObject()
{
	return collide_object_;
}

template <typename Scalar,int Dim>
ObjectBVH<Scalar, Dim>* RigidBodyArchive<Scalar, Dim>::objectBVH()
{
	return object_bvh_;
}

template <typename Scalar,int Dim>
RenderBase* RigidBodyArchive<Scalar, Dim>::render()
{
	return render_;
}


///////////////////////////////////////////////////////////////////////////////////////


template <typename Scalar,int Dim>
RigidBodyDriver<Scalar, Dim>::RigidBodyDriver():
	window_(NULL),
	scene_bvh_()
{

}

template <typename Scalar,int Dim>
RigidBodyDriver<Scalar, Dim>::~RigidBodyDriver()
{
	unsigned int num_rigid_body = numRigidBody();
	for(unsigned int i = 0; i < num_rigid_body; ++i)
	{
		delete rigid_body_archives_[i];
	}
	rigid_body_archives_.clear();
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::run()
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::advanceFrame()
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::initialize()
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::advanceStep(Scalar dt)
{
	collisionDetection();
}

template <typename Scalar,int Dim>
Scalar RigidBodyDriver<Scalar, Dim>::computeTimeStep()
{
	Scalar time_step = static_cast<Scalar>(0);
	return time_step;
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::write(const char *file_name)
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::read(const char *file_name)
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::addRigidBody(RigidBody<Scalar, Dim>* rigid_body, bool is_rebuild)
{
	RigidBodyArchive<Scalar, Dim>* archive = new RigidBodyArchive<Scalar, Dim>(rigid_body);
	archive->setIndex(numRigidBody());
	scene_bvh_.addObjectBVH(archive->objectBVH(), is_rebuild);
	rigid_body_archives_.push_back(archive);
	if(window_ != NULL)
		window_->pushBackRenderTask(archive->render());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::setWindow(GlutWindow* window)
{
	if(window == NULL)
		return;
	window_ = window;
	unsigned int num_rigid_body = numRigidBody();
	for(unsigned int i = 0; i < num_rigid_body; ++i)
	{
		window_->pushBackRenderTask(rigid_body_archives_[i]->render());
	}
}

template <typename Scalar,int Dim>
unsigned int RigidBodyDriver<Scalar, Dim>::numRigidBody() const
{
	return static_cast<unsigned int>(rigid_body_archives_.size());
}

template <typename Scalar,int Dim>
bool RigidBodyDriver<Scalar, Dim>::collisionDetection()
{
	scene_bvh_.updateSceneBVH();
	return scene_bvh_.selfCollide(collision_result_);
}


//explicit instantiation
template class RigidBodyArchive<float, 3>;
template class RigidBodyArchive<double, 3>;
template class RigidBodyDriver<float, 3>;
template class RigidBodyDriver<double, 3>;


}