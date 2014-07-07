/*
 * @file rigid_body_driver.h 
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_H_

#include<vector>
#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Geometry/Bounding_Volume/bvh_base.h"
#include "Physika_Geometry/Bounding_Volume/scene_bvh.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Dynamics/Collidable_Objects/contact_point.h"


namespace Physika{

template <typename Scalar,int Dim> class RigidBody;
template <typename Scalar,int Dim> class CollidableObject;
template <typename Scalar,int Dim> class BVHBase;
template <typename Scalar,int Dim> class ObjectBVH;

//Rigid body archive, which contains relative informations about a rigid body during simulation, e.g. the collidable object constructed from this rigid body and its BVH.
//This should be used as a internal class of RigidBodyDriver. It should be transparent to the outside.
template <typename Scalar,int Dim>
class RigidBodyArchive
{
public:
	RigidBodyArchive();
	RigidBodyArchive(RigidBody<Scalar, Dim>* rigid_body);
	virtual ~RigidBodyArchive();

	//get & set
	void setRigidBody(RigidBody<Scalar, Dim>* rigid_body);
	unsigned int index() const;
	void setIndex(unsigned int index);
	RigidBody<Scalar, Dim>* rigidBody();
	CollidableObject<Scalar, Dim>* collideObject();
	ObjectBVH<Scalar, Dim>* objectBVH();

protected:
	unsigned int index_;
	RigidBody<Scalar, Dim>* rigid_body_;
	CollidableObject<Scalar, Dim>* collide_object_;
	ObjectBVH<Scalar, Dim>* object_bvh_;
};

template <typename Scalar,int Dim> class RigidDriverPlugin;

template <typename Scalar,int Dim>
class RigidBodyDriver: public DriverBase<Scalar>
{
public:
	//constructors && deconstructors
	RigidBodyDriver();
	virtual ~RigidBodyDriver();

	//inherit functions
	void run();//run the simulation from start frame to end frame
	void advanceFrame();//advance one frame
	void initialize();//initialize before the simulation
	void advanceStep(Scalar dt);//advance one time step
	Scalar computeTimeStep();//compute time step with respect to simulation specific conditions
	void write(const char *file_name);//write simulation data to file
	void read(const char *file_name);//read simulation data from file

	//get & set, add & delete
	virtual void addRigidBody(RigidBody<Scalar, Dim>* rigid_body, bool is_rebuild = true);//is_rebuild means whether rebuild the scene BVH after adding this body.
	unsigned int numRigidBody() const;
	RigidBody<Scalar, Dim>* rigidBody(unsigned int index);
	CollisionDetectionResult<Scalar, Dim>& collisionResult();
    ContactPointManager<Scalar, Dim>& contactPoints();

	//dynamics
    virtual void updateRigidBody(Scalar dt);
	virtual bool collisionDetection();
    virtual void collisionResponse();

	//plugin
	void addPlugin(DriverPluginBase<Scalar>* plugin);

protected:
	SceneBVH<Scalar, Dim> scene_bvh_;
	std::vector<RigidBodyArchive<Scalar, Dim>* > rigid_body_archives_;
    CollisionDetectionResult<Scalar, Dim> collision_result_;
    ContactPointManager<Scalar, Dim> contact_points_;
    int frame_;
    int step_;
};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_H_