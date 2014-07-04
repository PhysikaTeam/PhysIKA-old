/*
 * @file rigid_body.cpp 
 * @Basic rigid_body class.
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
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::RigidBody():
	object_type_(CollidableObject<Scalar, Dim>::MESH_BASED),
	mesh_(NULL),
	transform_(),
    inertia_tensor_(),
    density_(1),
	is_fixed_(false)
{

}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::RigidBody(SurfaceMesh<Scalar>* mesh, Scalar density = 1)
{
    setProperty(mesh, density);
}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::RigidBody(SurfaceMesh<Scalar>* mesh, Transform<Scalar>& transform, Scalar density = 1)
{
    setProperty(mesh, transform, density);
}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::RigidBody(RigidBody<Scalar, Dim>& rigid_body)
{
    copy(rigid_body);
}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::~RigidBody()
{

}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::copy(RigidBody<Scalar, Dim>& rigid_body)
{
    object_type_ = rigid_body.object_type_;
    mesh_ = rigid_body.mesh_;
    transform_ = rigid_body.transform_;
    inertia_tensor_ = rigid_body.inertia_tensor_;
    density_ = rigid_body.density_;
    is_fixed_ = rigid_body.is_fixed_;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setTranslation(Vector<Scalar, 3>& translation)//Only defined to 3-Dimension
{
    transform_.setTranslation(translation);
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setRotation(Vector<Scalar, 3>& rotation)//Only defined to 3-Dimension
{
    //transform_.setRotation(rotation);
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setRotation(Quaternion<Scalar>& rotation)//Only defined to 3-Dimension
{
    transform_.setRotation(rotation);
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setRotation(SquareMatrix<Scalar, 3>& rotation)//Only defined to 3-Dimension
{
    //transform_.setRotation(rotation);
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setScale(Vector<Scalar, 3>& scale)//Only defined to 3-Dimension. Inertia tensor will be recalculated
{
    //transform_.setScale(scale);
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setProperty(SurfaceMesh<Scalar>* mesh, Scalar density)
{
    mesh_ = mesh;
    density_ = density;
    object_type_ = CollidableObject<Scalar, Dim>::MESH_BASED;
    inertia_tensor_.setBody(mesh_, Vector<Scalar, 3>(1, 1, 1), density_, local_mass_center_, mass_);//wait for scale in transform
    
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setProperty(SurfaceMesh<Scalar>* mesh, Transform<Scalar>& transform, Scalar density)
{
    mesh_ = mesh;
    transform_ = transform;
    density_ = density;
    object_type_ = CollidableObject<Scalar, Dim>::MESH_BASED;
    //inertia_tensor_.setBody(mesh_, transform_.scalar, density_, local_mass_center_, mass_);//wait for scale in transform
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::update(Scalar dt)
{
    resetTemporaryVariables();
    velocityIntegral(dt);
    configurationIntegral(dt);
    updateInertiaTensor();
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::addImpulse(Scalar magnitude, const Vector<Scalar, Dim>& direction, const Vector<Scalar, Dim>& global_position)
{
    Vector<Scalar, Dim> impulse = direction * magnitude;
    global_translation_impulse_ += impulse;
    global_angular_velocity_ += (global_position - globalMassCenter()).cross(impulse);
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> RigidBody<Scalar, Dim>::globalVertexPosition(unsigned int vertex_idnex) const
{
    Vector<Scalar, Dim> local_position = mesh_->vertexPosition(vertex_idnex);
    return transform_.transform(local_position);
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> RigidBody<Scalar, Dim>::globalVertexVelocity(unsigned int vertex_index) const
{
     return globalPointVelocity(globalVertexPosition(vertex_index));
}

template <typename Scalar,int Dim>
Vector<Scalar, Dim> RigidBody<Scalar, Dim>::globalPointVelocity(const Vector<Scalar, Dim>& global_point_position) const
{
     Vector<Scalar, Dim> velocity = global_translation_velocity_;
     velocity += global_angular_velocity_.cross(global_point_position - globalMassCenter());
     return velocity;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::resetTemporaryVariables()
{
    global_translation_impulse_ *= 0;
    global_angular_impulse_ *= 0;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::velocityIntegral(Scalar dt)
{
    global_translation_velocity_ += global_translation_impulse_ / mass_;
    global_angular_velocity_ += spatialInertiaTensorInverse() * global_angular_impulse_;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::configurationIntegral(Scalar dt)
{
    global_translation_ += global_translation_velocity_ * dt;
    //to do: rotation
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::updateInertiaTensor()
{
    inertia_tensor_.rotate(global_rotation_);
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setMesh(SurfaceMesh<Scalar>* mesh)
{
    mesh_ = mesh;
    object_type_ = CollidableObject<Scalar, Dim>::MESH_BASED;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setTransform(Transform<Scalar>& transform)
{
    transform_ = transform;
}


//explicit instantiation
template class RigidBody<float, 3>;
template class RigidBody<double, 3>;

} //end of namespace Physika
