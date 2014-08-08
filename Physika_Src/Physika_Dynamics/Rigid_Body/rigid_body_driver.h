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

#include <string>
#include <vector>
#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_method.h"
#include "Physika_Dynamics/Rigid_Body/rigid_response_method.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Utilities/dimension_trait.h"


namespace Physika{

template <typename Scalar,int Dim> class RigidBody;
template <typename Scalar,int Dim> class CollidableObject;

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
	void setRigidBody(RigidBody<Scalar, Dim>* rigid_body);//This function will do nothing but call the overload version depending on the dimension
	unsigned int index() const;
	void setIndex(unsigned int index);
	RigidBody<Scalar, Dim>* rigidBody();
	CollidableObject<Scalar, Dim>* collideObject();

protected:
	unsigned int index_;
	RigidBody<Scalar, Dim>* rigid_body_;
	CollidableObject<Scalar, Dim>* collide_object_;

    //Overload versions of utilities for 2D and 3D situations
    void setRigidBody(RigidBody<Scalar, Dim>* rigid_body, DimensionTrait<2> trait);
    void setRigidBody(RigidBody<Scalar, Dim>* rigid_body, DimensionTrait<3> trait);
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
    void initConfiguration(const std::string &file_name);
    void printConfigFileFormat();
    void initSimulationData();
	void advanceStep(Scalar dt);//advance one time step
	Scalar computeTimeStep();//compute time step with respect to simulation specific conditions
    bool withRestartSupport() const; //indicate wheter restart is supported in implementation
	void write(const std::string &file_name);//write simulation data to file
	void read(const std::string &file_name);//read simulation data from file

	//get & set, add & delete
	virtual void addRigidBody(RigidBody<Scalar, Dim>* rigid_body);
    void setGravity(Scalar gravity);//gravity is along the y-axis and positive value means -y direction. Gravity is usually set to 9.81
	unsigned int numRigidBody() const;
	RigidBody<Scalar, Dim>* rigidBody(unsigned int index);
    unsigned int numCollisionPair() const;
    CollisionPairBase<Scalar, Dim>* collisionPair(unsigned int index);
    unsigned int numContactPoint() const;
    ContactPoint<Scalar, Dim>* contactPoint(unsigned int index);

	//plugin
	void addPlugin(DriverPluginBase<Scalar>* plugin);

protected:
    //dynamics
    virtual void performGravity(Scalar dt);
    virtual bool collisionDetection();
    virtual void collisionResponse();
    virtual void updateRigidBody(Scalar dt);

	std::vector<RigidBodyArchive<Scalar, Dim>* > rigid_body_archives_;
    CollisionDetectionMethod<Scalar, Dim>* collision_detection_method_;
    RigidResponseMethod<Scalar, Dim>* collision_response_method_;
    Scalar gravity_;
    unsigned int step_;

};

} //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_H_
