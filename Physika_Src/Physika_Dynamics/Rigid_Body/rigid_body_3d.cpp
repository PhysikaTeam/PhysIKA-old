/*
 * @file rigid_body_3d.cpp 
 * @3D rigid_body class.
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
#include "Physika_Dynamics/Rigid_Body/rigid_body_3d.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar>
RigidBody<Scalar, 3>::RigidBody():
object_type_(CollidableObjectInternal::MESH_BASED),
    mesh_(NULL),
    transform_(),
    inertia_tensor_(),
    density_(1),
    is_fixed_(false),
    coeff_restitution_(1),
    coeff_friction_(0),
    global_translation_(0),
    global_rotation_(),
    global_translation_velocity_(0),
    global_angular_velocity_(0),
    global_translation_impulse_(0),
    global_angular_impulse_(0)
{

}

template <typename Scalar>
RigidBody<Scalar, 3>::RigidBody(SurfaceMesh<Scalar>* mesh, Scalar density):
object_type_(CollidableObjectInternal::MESH_BASED),
    transform_(),
    inertia_tensor_(),
    is_fixed_(false),
    coeff_restitution_(1),
    coeff_friction_(0),
    global_translation_(0),
    global_rotation_(),
    global_translation_velocity_(0),
    global_angular_velocity_(0),
    global_translation_impulse_(0),
    global_angular_impulse_(0)
{
    setProperty(mesh, density);
}

template <typename Scalar>
RigidBody<Scalar, 3>::RigidBody(SurfaceMesh<Scalar>* mesh, const Transform<Scalar>& transform, Scalar density):
object_type_(CollidableObjectInternal::MESH_BASED),
    inertia_tensor_(),
    is_fixed_(false),
    coeff_restitution_(1),
    coeff_friction_(0),
    global_translation_(0),
    global_rotation_(),
    global_translation_velocity_(0),
    global_angular_velocity_(0),
    global_translation_impulse_(0),
    global_angular_impulse_(0)
{
    setProperty(mesh, transform, density);
}

template <typename Scalar>
RigidBody<Scalar, 3>::RigidBody(const RigidBody<Scalar, 3>& rigid_body)
{
    copy(rigid_body);
}

template <typename Scalar>
RigidBody<Scalar, 3>::~RigidBody()
{

}

template <typename Scalar>
void RigidBody<Scalar, 3>::copy(const RigidBody<Scalar, 3>& rigid_body)
{
    object_type_ = rigid_body.object_type_;
    mesh_ = rigid_body.mesh_;
    transform_ = rigid_body.transform_;
    density_ = rigid_body.density_;
    is_fixed_ = rigid_body.is_fixed_;
    coeff_restitution_ = rigid_body.coeff_restitution_;
    coeff_friction_ = rigid_body.coeff_friction_;

    inertia_tensor_ = rigid_body.inertia_tensor_;
    mass_ = rigid_body.mass_;
    local_mass_center_ = rigid_body.local_mass_center_;

    global_translation_ = rigid_body.global_translation_;
    global_rotation_ = rigid_body.global_rotation_;
    global_translation_velocity_ = rigid_body.global_translation_velocity_;
    global_angular_velocity_ = rigid_body.global_angular_velocity_;
    global_translation_impulse_ = rigid_body.global_translation_impulse_;
    global_angular_impulse_ = rigid_body.global_angular_impulse_;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setTranslation(const Vector<Scalar, 3>& translation)//Only defined to 3-Dimension
{
    transform_.setTranslation(translation);
    recalculatePosition();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setRotation(const Vector<Scalar, 3>& rotation)//Only defined to 3-Dimension
{
    transform_.setRotation(rotation);
    recalculatePosition();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setRotation(const Quaternion<Scalar>& rotation)//Only defined to 3-Dimension
{
    transform_.setRotation(rotation);
    recalculatePosition();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setRotation(const SquareMatrix<Scalar, 3>& rotation)//Only defined to 3-Dimension
{
    transform_.setRotation(rotation);
    recalculatePosition();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setScale(const Vector<Scalar, 3>& scale)//Only defined to 3-Dimension. Inertia tensor will be recalculated
{
    transform_.setScale(scale);
    inertia_tensor_.setBody(mesh_, transform_.scale(), density_, local_mass_center_, mass_);
    recalculatePosition();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setProperty(SurfaceMesh<Scalar>* mesh, Scalar density)
{
    mesh_ = mesh;
    density_ = density;
    object_type_ = CollidableObjectInternal::MESH_BASED;
    inertia_tensor_.setBody(mesh_, transform_.scale(), density_, local_mass_center_, mass_);
    recalculatePosition();   
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setProperty(SurfaceMesh<Scalar>* mesh, const Transform<Scalar>& transform, Scalar density)
{
    mesh_ = mesh;
    transform_ = transform;
    density_ = density;
    object_type_ = CollidableObjectInternal::MESH_BASED;
    inertia_tensor_.setBody(mesh_, transform_.scale(), density_, local_mass_center_, mass_);
    recalculatePosition();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::update(Scalar dt)
{
    velocityIntegral(dt);
    configurationIntegral(dt);
    updateInertiaTensor();
    recalculateTransform();
    resetTemporaryVariables();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::addImpulse(Scalar magnitude, const Vector<Scalar, 3>& direction, const Vector<Scalar, 3>& global_position)
{
    Vector<Scalar, 3> impulse = direction * magnitude;
    global_translation_impulse_ += impulse;
    global_angular_impulse_ += (global_position - globalMassCenter()).cross(impulse);
}

template <typename Scalar>
void RigidBody<Scalar, 3>::addTranslationImpulse(const Vector<Scalar, 3>& impulse)
{
    global_translation_impulse_ += impulse;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::addAngularImpulse(const Vector<Scalar, 3>& impulse)
{
    global_angular_impulse_ += impulse;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::performGravity(Scalar gravity, Scalar dt)
{
    if(is_fixed_)
        return;
    global_translation_velocity_[1] -= gravity * dt;
}

template <typename Scalar>
Vector<Scalar, 3> RigidBody<Scalar, 3>::globalVertexPosition(unsigned int vertex_idnex) const
{
    Vector<Scalar, 3> local_position = mesh_->vertexPosition(vertex_idnex);
    return transform_.transform(local_position);
}

template <typename Scalar>
Vector<Scalar, 3> RigidBody<Scalar, 3>::globalVertexVelocity(unsigned int vertex_index) const
{
    return globalPointVelocity(globalVertexPosition(vertex_index));
}

template <typename Scalar>
Vector<Scalar, 3> RigidBody<Scalar, 3>::globalPointVelocity(const Vector<Scalar, 3>& global_point_position) const
{
    Vector<Scalar, 3> velocity = global_translation_velocity_;
    velocity += global_angular_velocity_.cross(global_point_position - globalMassCenter());
    return velocity;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::resetTemporaryVariables()
{
    global_translation_impulse_ *= 0;
    global_angular_impulse_ *= 0;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::velocityIntegral(Scalar dt)
{
    if(is_fixed_)
        return;
    global_translation_velocity_ += global_translation_impulse_ / mass_;
    global_angular_velocity_ += spatialInertiaTensorInverse() * global_angular_impulse_;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::configurationIntegral(Scalar dt)
{
    if(is_fixed_)
        return;
    global_translation_ += global_translation_velocity_ * dt;
    Quaternion<Scalar> quad;
    quad.setX(global_angular_velocity_[0]);
    quad.setY(global_angular_velocity_[1]);
    quad.setZ(global_angular_velocity_[2]);
    quad.setW(0);
    quad = quad * global_rotation_ / 2;
    global_rotation_ += quad * dt;
    global_rotation_.normalize();
}

template <typename Scalar>
void RigidBody<Scalar, 3>::updateInertiaTensor()
{
    inertia_tensor_.rotate(global_rotation_);
}

template <typename Scalar>
void RigidBody<Scalar, 3>::recalculateTransform()
{
    transform_.setRotation(global_rotation_);
    transform_.setTranslation(global_translation_ - global_rotation_.rotate(local_mass_center_));
}

template <typename Scalar>
void RigidBody<Scalar, 3>::recalculatePosition()
{
    global_rotation_ = transform_.rotation();
    global_translation_ = transform_.translation() + global_rotation_.rotate(local_mass_center_);
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setMesh(SurfaceMesh<Scalar>* mesh)
{
    mesh_ = mesh;
    object_type_ = CollidableObjectInternal::MESH_BASED;
}

template <typename Scalar>
void RigidBody<Scalar, 3>::setTransform(Transform<Scalar>& transform)
{
    transform_ = transform;
}


//explicit instantiation
template class RigidBody<float, 3>;
template class RigidBody<double, 3>;

} //end of namespace Physika