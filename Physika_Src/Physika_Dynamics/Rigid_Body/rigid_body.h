/*
 * @file rigid_body.h 
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_H_

#include "Physika_Core/Transform/transform.h"
#include "Physika_Dynamics/Rigid_Body/inertia_tensor.h"

namespace Physika{

template <typename Scalar,int Dim> class CollidableObject;
template <typename Scalar,int Dim> class Vector;
template <typename Scalar> class SurfaceMesh;

//RigidBody only supports 3-Dimension rigid body for now
//We strongly recommend using copy construction function to construct a rigid body from another one with the same mesh, scale and density because inertia tensor will not be recalculated.
template <typename Scalar,int Dim>
class RigidBody
{
public:
	//constructors && deconstructors
	RigidBody();
    RigidBody(SurfaceMesh<Scalar>* mesh, Scalar density = 1);
    RigidBody(SurfaceMesh<Scalar>* mesh, Transform<Scalar>& transform, Scalar density = 1);
    RigidBody(RigidBody<Scalar, Dim>& rigid_body);
	virtual ~RigidBody();

	//get & set
    void copy(RigidBody<Scalar, Dim>& rigid_body);//Using this function for construction is strongly recommended because inertia tensor will not be recalculated for the same mesh.
    inline typename CollidableObject<Scalar, Dim>::ObjectType objectType() const {return object_type_;};
    inline SurfaceMesh<Scalar>* mesh() {return mesh_;};
    inline Transform<Scalar> transform() const {return transform_;};
    inline Transform<Scalar> transform() {return transform_;};
	inline const Transform<Scalar>* transformPtr() const {return &transform_;};
	inline Transform<Scalar>* transformPtr() {return &transform_;};//WARNING! Don't use this to modify the transform of this rigid body. Use setTranslate(), setRotate() and setScale() instead.
    void setTranslation(Vector<Scalar, 3>& translation);//Only defined to 3-Dimension
    void setRotation(Vector<Scalar, 3>& rotation);//Only defined to 3-Dimension
    void setRotation(Quaternion<Scalar>& rotation);//Only defined to 3-Dimension
    void setRotation(SquareMatrix<Scalar, 3>& rotation);//Only defined to 3-Dimension
    void setScale(Vector<Scalar, 3>& scale);//Only defined to 3-Dimension. Inertia tensor will be recalculated
    void setProperty(SurfaceMesh<Scalar>* mesh, Scalar density = 1);//Inertia tensor will be recalculated
    void setProperty(SurfaceMesh<Scalar>* mesh, Transform<Scalar>& transform, Scalar density = 1);//Inertia tensor will be recalculated
    inline void setFixed(bool is_fixed) {is_fixed_ = is_fixed;};
    inline bool isFixed() const {return is_fixed_;};
    inline const SquareMatrix<Scalar, 3> spatialInertiaTensor() const {return inertia_tensor_.spatialInertiaTensor();};
    inline const SquareMatrix<Scalar, 3> bodyInertiaTensor() const {return inertia_tensor_.bodyInertiaTensor();};
    inline const SquareMatrix<Scalar, 3> spatialInertiaTensorInverse() const {return inertia_tensor_.spatialInertiaTensorInverse();};
    inline const SquareMatrix<Scalar, 3> bodyInertiaTensorInverse() const {return inertia_tensor_.bodyInertiaTensorInverse();};
    inline Scalar density() const {return density_;};
    inline Scalar mass() const {return mass_;};
    inline Vector<Scalar, Dim> localMassCenter() const {return local_mass_center_;};
    inline Vector<Scalar, Dim> globalMassCenter() const {return transform_.rotation().rotate(local_mass_center_) + transform_.translation();};//Can't use transform_.transform() here because its scales local_mass_center_ unexpectedly
    inline Vector<Scalar, Dim> globalTranslation() const {return global_translation_;};
    inline Quaternion<Scalar> globalRotation() const {return global_rotation_;};
    inline Vector<Scalar, Dim> globalTranslationVelocity() const {return global_translation_velocity_;};
    inline Vector<Scalar, Dim> globalAngularVelocity() const {return global_angular_velocity_;};
    inline void setGlobalTranslationVelocity(const Vector<Scalar, Dim>& velocity) {global_translation_velocity_ = velocity;};
    inline void setGlobalAngularVelocity(const Vector<Scalar, Dim>& velocity) {global_angular_velocity_ = velocity;};

	//dynamics
	void update(Scalar dt);//update its configuration and velocity
    void addImpulse(Scalar magnitude, const Vector<Scalar, Dim>& direction, const Vector<Scalar, Dim>& global_position);//accumulate collision impulse to the rigid body. This will not change its velocity until velocityIntegral has been called

    Vector<Scalar, Dim> globalVertexPosition(unsigned int vertex_idnex) const;//get the position of a vertex in global frame
    Vector<Scalar, Dim> globalVertexVelocity(unsigned int vertex_index) const;//get the velocity of a vertex in global frame
    Vector<Scalar, Dim> globalPointVelocity(const Vector<Scalar, Dim>& global_point_position) const;//get the velocity of an arbitrary point on/inside the rigid body in global frame

protected:
	//basic properties of a rigid body
    
    //can be set by public functions
	typename CollidableObject<Scalar, Dim>::ObjectType object_type_;
	SurfaceMesh<Scalar>* mesh_;
	Transform<Scalar> transform_;
	Scalar density_;
	bool is_fixed_;

    //obtained by internal computation
    InertiaTensor<Scalar> inertia_tensor_;
    Scalar mass_;
    Vector<Scalar, Dim> local_mass_center_;//position of mass center in local frame (mesh frame)

    //configuration
    Vector<Scalar, Dim> global_translation_;//translation of mass center in global frame (inertia frame). Different from translation in transform_ (which is the translation of local frame in global frame) 
    Quaternion<Scalar> global_rotation_;//rotation of rigid body in global frame (inertia frame). Same with rotation in transform_
    //velocity
    Vector<Scalar, Dim> global_translation_velocity_;//velocity of mass center in global frame (inertia frame)
    Vector<Scalar, Dim> global_angular_velocity_;//angular velocity of rigid body in global frame (inertia frame)
    //impulse
    Vector<Scalar, Dim> global_translation_impulse_;//translation impulse accumulated during the collision. This will be used to update global_translation_velocity_
    Vector<Scalar, Dim> global_angular_impulse_;///rotation impulse accumulated during the collision. This will be used to update global_angular_velocity_

    //Internal functions

    //dynamics
    void resetTemporaryVariables();//prepare for the new time step
    void velocityIntegral(Scalar dt);//Only defined to 3-Dimension
    void configurationIntegral(Scalar dt);
    void updateInertiaTensor();
    void recalculateTransform();//recalculate transform_ from global_translation_ and global_rotation_
    void recalculatePosition();//recalculate global_translation_ and global_rotation_ from transform_

    //set
    void setMesh(SurfaceMesh<Scalar>* mesh);
    void setTransform(Transform<Scalar>& transform);
	
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_H_
