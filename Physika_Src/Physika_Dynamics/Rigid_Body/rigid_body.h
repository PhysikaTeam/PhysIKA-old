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
	typename CollidableObject<Scalar, Dim>::ObjectType objectType() const;
	SurfaceMesh<Scalar>* mesh();
	Transform<Scalar> transform() const;
	Transform<Scalar> transform();
	const Transform<Scalar>* transformPtr() const;
	Transform<Scalar>* transformPtr();//WARNING! Don't use this to modify the transform of this rigid body. Use setTranslate(), setRotate() and setScale() instead.
    void setTranslation(Vector<Scalar, 3>& translation);//Only defined to 3-Dimension
    void setRotation(Vector<Scalar, 3>& rotation);//Only defined to 3-Dimension
    void setRotation(Quaternion<Scalar>& rotation);//Only defined to 3-Dimension
    void setRotation(SquareMatrix<Scalar, 3>& rotation);//Only defined to 3-Dimension
    void setScale(Vector<Scalar, 3>& scale);//Only defined to 3-Dimension. Inertia tensor will be recalculated
    void setProperty(SurfaceMesh<Scalar>* mesh, Scalar density = 1);//Inertia tensor will be recalculated
    void setProperty(SurfaceMesh<Scalar>* mesh, Transform<Scalar>& transform, Scalar density = 1);//Inertia tensor will be recalculated
    void setFixed(bool is_fixed);
    bool isFixed() const;
    const SquareMatrix<Scalar, 3> spatialInertiaTensor() const;
    const SquareMatrix<Scalar, 3> bodyInertiaTensor() const;
    Scalar density() const;
    Scalar mass() const;
    Vector<Scalar, Dim> localMassCenter() const;

	//dynamics
	void update(Scalar dt);//update its configuration and velocity

    Vector<Scalar, Dim> globalVertexPosition(unsigned int vertex_idnex) const;//get the position of a vertex in global frame
    Vector<Scalar, Dim> globalVertexVelocity(unsigned int vertex_index) const;//get the velocity of a vertex in global frame
    Vector<Scalar, Dim> globalPointVelocity(Vector<Scalar, Dim> point_position) const;//get the velocity of an arbitrary point on/inside the rigid body in global frame

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
    Vector<Scalar, Dim> global_translation_velocity;//velocity of mass center in global frame (inertia frame)
    Vector<Scalar, Dim> global_angular_velocity;//angular velocity of rigid body in global frame (inertia frame)

    //Internal functions

    //set
    void setMesh(SurfaceMesh<Scalar>* mesh);
    void setTransform(Transform<Scalar>& transform);

    //dynamics
    void velocityIntegral(Scalar dt);
    void configurationIntegral(Scalar dt);
    void updateInertiaTensor();
	
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_H_
