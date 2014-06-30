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
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"

namespace Physika{

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::RigidBody():
	object_type_(CollidableObject<Scalar, Dim>::MESH_BASED),
	mesh_(NULL),
	transform_(),
	mass_(1),
	is_fixed_(false)
{

}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>::~RigidBody()
{

}
template <typename Scalar,int Dim>
typename CollidableObject<Scalar, Dim>::ObjectType RigidBody<Scalar, Dim>::objectType() const
{
	return object_type_;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setMesh(SurfaceMesh<Scalar>* mesh)
{
	mesh_ = mesh;
	object_type_ = CollidableObject<Scalar, Dim>::MESH_BASED;
}

template <typename Scalar,int Dim>
SurfaceMesh<Scalar>* RigidBody<Scalar, Dim>::mesh()
{
	return mesh_;
}

template <typename Scalar,int Dim>
void RigidBody<Scalar, Dim>::setTransform(Transform<Scalar>& transform)
{
	transform_ = transform;
}

template <typename Scalar,int Dim>
Transform<Scalar> RigidBody<Scalar, Dim>::transform() const
{
	return transform_;
}

template <typename Scalar,int Dim>
Transform<Scalar> RigidBody<Scalar, Dim>::transform()
{
	return transform_;
}

template <typename Scalar,int Dim>
const Transform<Scalar>* RigidBody<Scalar, Dim>::transformPtr() const
{
	return &transform_;
}

template <typename Scalar,int Dim>
Transform<Scalar>* RigidBody<Scalar, Dim>::transformPtr()
{
	return &transform_;
}

//explicit instantiation
template class RigidBody<float, 3>;
template class RigidBody<double, 3>;

} //end of namespace Physika
