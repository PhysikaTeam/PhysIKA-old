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

#include <limits>
#include "Physika_Dynamics/Rigid_Body/rigid_body.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_2d.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_3d.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver_utility.h"
#include "Physika_Geometry/Bounding_Volume/object_bvh.h"
#include "Physika_Dynamics/Collidable_Objects/mesh_based_collidable_object.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Collidable_Objects/collision_detection_result.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"

namespace Physika{

///////////////////////////////////////////////////////////////////////////////////////
//RigidBodyArchive
///////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar,int Dim>
RigidBodyArchive<Scalar, Dim>::RigidBodyArchive():
	index_(0),
	rigid_body_(NULL),
	collide_object_(NULL),
	object_bvh_(NULL)
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
}

template <typename Scalar,int Dim>
void RigidBodyArchive<Scalar, Dim>::setRigidBody(RigidBody<Scalar, Dim>* rigid_body)
{
    setRigidBody(rigid_body, DimensionTrait<Dim>());
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
void RigidBodyArchive<Scalar, Dim>::setRigidBody(RigidBody<Scalar, Dim>* rigid_body, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar,int Dim>
void RigidBodyArchive<Scalar, Dim>::setRigidBody(RigidBody<Scalar, Dim>* rigid_body, DimensionTrait<3> trait)
{
    if(rigid_body == NULL)
        return;

    rigid_body_ = rigid_body;
    RigidBody<Scalar, 3>* rigid_body_3d = dynamic_cast<RigidBody<Scalar, 3>* >(rigid_body);

    switch(rigid_body->objectType())
    {
    case CollidableObjectInternal::MESH_BASED: collide_object_ = dynamic_cast<CollidableObject<Scalar, Dim>* >(new MeshBasedCollidableObject<Scalar>());break;
    default: std::cerr<<"Object type error!"<<std::endl; return;
    }
    MeshBasedCollidableObject<Scalar>* mesh_object = dynamic_cast<MeshBasedCollidableObject<Scalar>*>(collide_object_);
    mesh_object->setMesh(rigid_body_3d->mesh());
    mesh_object->setTransform(rigid_body_3d->transformPtr());

    object_bvh_ = new ObjectBVH<Scalar, Dim>();
    object_bvh_->setCollidableObject(collide_object_);
}


///////////////////////////////////////////////////////////////////////////////////////
//RigidBodyDriver
///////////////////////////////////////////////////////////////////////////////////////


template <typename Scalar,int Dim>
RigidBodyDriver<Scalar, Dim>::RigidBodyDriver():
	scene_bvh_(),
    gravity_(9.81),
    time_step_(0.01),
    frame_(0),
    step_(0)
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
void RigidBodyDriver<Scalar, Dim>::initConfiguration(const std::string &file_name)
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::run()
{

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::advanceFrame()
{
    //update frame
    frame_++;

    //plugin
	unsigned int plugin_num = static_cast<unsigned int>((this->plugins_).size());
	RigidDriverPlugin<Scalar, Dim>* plugin;
	for(unsigned int i = 0; i < plugin_num; ++i)
	{
		plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
		if(plugin != NULL)
			plugin->onBeginFrame(frame_);
	}

    //plugin
    for(unsigned int i = 0; i < plugin_num; ++i)
    {
        plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
        if(plugin != NULL)
            plugin->onEndFrame(frame_);
    }
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::advanceStep(Scalar dt)
{
    //update step
    step_++;

    //plugin
    unsigned int plugin_num = static_cast<unsigned int>((this->plugins_).size());
    RigidDriverPlugin<Scalar, Dim>* plugin;
    for(unsigned int i = 0; i < plugin_num; ++i)
    {
        plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
        if(plugin != NULL)
            plugin->onBeginRigidStep(step_, dt);
    }

    //simulation step
    performGravity(dt);
	collisionDetection();
    collisionResponse();
    updateRigidBody(dt);

    //plugin
    for(unsigned int i = 0; i < plugin_num; ++i)
    {
        plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
        if(plugin != NULL)
            plugin->onEndRigidStep(step_, dt);
    }
}

template <typename Scalar,int Dim>
Scalar RigidBodyDriver<Scalar, Dim>::computeTimeStep()
{
	return time_step_;
}

template <typename Scalar,int Dim>
bool RigidBodyDriver<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::write(const std::string &file_name)
{
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::read(const std::string &file_name)
{
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::addRigidBody(RigidBody<Scalar, Dim>* rigid_body, bool is_rebuild)
{
	if(rigid_body == NULL)
		return;

    //add this rigid body
	RigidBodyArchive<Scalar, Dim>* archive = new RigidBodyArchive<Scalar, Dim>(rigid_body);
	archive->setIndex(numRigidBody());
	scene_bvh_.addObjectBVH(archive->objectBVH(), is_rebuild);
	rigid_body_archives_.push_back(archive);

    //plugin
	unsigned int plugin_num = static_cast<unsigned int>((this->plugins_).size());
	RigidDriverPlugin<Scalar, Dim>* plugin;
	for(unsigned int i = 0; i < plugin_num; ++i)
	{
		plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
		if(plugin != NULL)
			plugin->onAddRigidBody(rigid_body);
	}
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::setGravity(Scalar gravity)
{
    gravity_ = gravity;
}

template <typename Scalar,int Dim>
unsigned int RigidBodyDriver<Scalar, Dim>::numRigidBody() const
{
	return static_cast<unsigned int>(rigid_body_archives_.size());
}

template <typename Scalar,int Dim>
RigidBody<Scalar, Dim>* RigidBodyDriver<Scalar, Dim>::rigidBody(unsigned int index)
{
	if(index >= numRigidBody())
	{
		std::cerr<<"Rigid body index out of range!"<<std::endl;
		return NULL;
	}
	return rigid_body_archives_[index]->rigidBody();
}

template <typename Scalar,int Dim>
CollisionDetectionResult<Scalar, Dim>& RigidBodyDriver<Scalar, Dim>::collisionResult()
{
	return collision_result_;
}

template <typename Scalar,int Dim>
ContactPointManager<Scalar, Dim>& RigidBodyDriver<Scalar, Dim>::contactPoints()
{
    return contact_points_;
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::addPlugin(DriverPluginBase<Scalar>* plugin)
{
    if(plugin == NULL)
    {
        std::cerr<<"Null plugin!"<<std::endl;
        return;
    }
    if(dynamic_cast<RigidDriverPlugin<Scalar, Dim>* >(plugin) == NULL)
    {
        std::cerr<<"Wrong plugin type!"<<std::endl;
        return;
    }
    plugin->setDriver(this);
    this->plugins_.push_back(plugin);
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::initialize()
{
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::performGravity(Scalar dt)
{
    unsigned int num_rigid_body = numRigidBody();
    RigidBody<Scalar, Dim>* rigid_body;
    for(unsigned int i = 0; i < num_rigid_body; i++)
    {
        rigid_body = rigid_body_archives_[i]->rigidBody();
        if(!rigid_body->isFixed())
        {
            rigid_body->performGravity(gravity_, dt);
        }
    }
}

template <typename Scalar,int Dim>
bool RigidBodyDriver<Scalar, Dim>::collisionDetection()
{
    //plugin
    unsigned int plugin_num = static_cast<unsigned int>((this->plugins_).size());
    RigidDriverPlugin<Scalar, Dim>* plugin;
    for(unsigned int i = 0; i < plugin_num; ++i)
    {
        plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
        if(plugin != NULL)
            plugin->onBeginCollisionDetection();
    }

    //clean
    collision_result_.resetCollisionResults();
    contact_points_.cleanContactPoints();

    //update and collide
    scene_bvh_.updateSceneBVH();
    bool is_collide = scene_bvh_.selfCollide(collision_result_);
    contact_points_.setCollisionResult(collision_result_);

    //plugin
    plugin_num = static_cast<unsigned int>((this->plugins_).size());
    for(unsigned int i = 0; i < plugin_num; ++i)
    {
        plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
        if(plugin != NULL)
            plugin->onEndCollisionDetection();
    }
    return is_collide;
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::collisionResponse()
{
    //initialize
    unsigned int m = contact_points_.numContactPoint();//m: number of contact points
    unsigned int n = numRigidBody();//n: number of rigid bodies
    if(m == 0 || n == 0)//no collision or no rigid body
        return;
    unsigned int six_n = n * 6;//six_n: designed only for 3-dimension rigid bodies. The DoF(Degree of Freedom) of a rigid-body system
    unsigned int fric_sample_count = 2;//count of friction sample directions
    unsigned int s = m * fric_sample_count;//s: number of friction sample. Here a square sample is adopted
    SparseMatrix<Scalar> J(m, six_n);//Jacobian matrix
    SparseMatrix<Scalar> M_inv(six_n, six_n);//inversed inertia matrix
    SparseMatrix<Scalar> D(s, six_n);//Jacobian matrix of friction
    SparseMatrix<Scalar> JMJ(m, m);
    SparseMatrix<Scalar> JMD(m, s);
    SparseMatrix<Scalar> DMJ(s, m);
    SparseMatrix<Scalar> DMD(s, s);
    VectorND<Scalar> v(six_n, 0);//generalized velocity of the system
    VectorND<Scalar> Jv(m, 0);//normal relative velocity of each contact point (for normal contact impulse calculation)
    VectorND<Scalar> Dv(s, 0);//tangent relative velocity of each contact point (for frictional contact impulse calculation)
    VectorND<Scalar> CoR(m, 0);//coefficient of restitution (for normal contact impulse calculation)
    VectorND<Scalar> CoF(s, 0);//coefficient of friction (for frictional contact impulse calculation)
    VectorND<Scalar> z_norm(m, 0);//normal contact impulse. The key of collision response
    VectorND<Scalar> z_fric(s, 0);//frictional contact impulse. The key of collision response

    //compute the matrix of dynamics
    computeInvMassMatrix(M_inv);
    computeJacobianMatrix(J);
    computeFricJacobianMatrix(D);
    computeGeneralizedVelocity(v);

    //compute other matrix in need
    SparseMatrix<Scalar> J_T = J;
    J_T = J.transpose();
    SparseMatrix<Scalar> D_T = D;
    D_T = D.transpose();
    SparseMatrix<Scalar> MJ = M_inv * J_T;
    SparseMatrix<Scalar> MD = M_inv * D_T;
    JMJ = J * MJ;
    DMD = D * MD;
    JMD = J * MD;
    DMJ = D * MJ;
    Jv = J * v;
    Dv = D * v;

    //update CoR and CoF
    computeCoefficient(CoR, CoF);

    //solve BLCP with PGS. z_norm and z_fric are the unknown variables
    solveBLCPWithPGS(JMJ, DMD, JMD, DMJ, Jv, Dv, z_norm, z_fric, CoR, CoF);

    //apply impulse
    applyImpulse(z_norm, z_fric, J_T, D_T);
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::updateRigidBody(Scalar dt)
{
    //update
    unsigned int num_rigid_body = numRigidBody();
    RigidBody<Scalar, Dim>* rigid_body;
    for(unsigned int i = 0; i < num_rigid_body; i++)
    {
        rigid_body = rigid_body_archives_[i]->rigidBody();
        if(!rigid_body->isFixed())
        {
            //rigid_body->addTranslationImpulse(gravity * rigid_body->mass());//We don't update the velocity and position of a fixed body.
            rigid_body->update(dt);
        }
    }
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::computeInvMassMatrix(SparseMatrix<Scalar>& M_inv)
{
    RigidBodyDriverUtility<Scalar>::computeInvMassMatrix(this, M_inv, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::computeJacobianMatrix(SparseMatrix<Scalar>& J)
{
    RigidBodyDriverUtility<Scalar>::computeJacobianMatrix(this, J, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::computeFricJacobianMatrix(SparseMatrix<Scalar>& D)
{
    RigidBodyDriverUtility<Scalar>::computeFricJacobianMatrix(this, D, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::computeGeneralizedVelocity(VectorND<Scalar>& v)
{
    RigidBodyDriverUtility<Scalar>::computeGeneralizedVelocity(this, v, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::computeCoefficient(VectorND<Scalar>& CoR, VectorND<Scalar>& CoF)
{
    RigidBodyDriverUtility<Scalar>::computeCoefficient(this, CoR, CoF, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::solveBLCPWithPGS(SparseMatrix<Scalar>& JMJ, SparseMatrix<Scalar>& DMD, SparseMatrix<Scalar>& JMD, SparseMatrix<Scalar>& DMJ,
    VectorND<Scalar>& Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
    VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count)
{
    RigidBodyDriverUtility<Scalar>::solveBLCPWithPGS(this, JMJ, DMD, JMD, DMJ, Jv, Dv, z_norm, z_fric, CoR, CoF, iteration_count, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::applyImpulse(VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric, SparseMatrix<Scalar>& J_T, SparseMatrix<Scalar>& D_T)
{
    RigidBodyDriverUtility<Scalar>::applyImpulse(this, z_norm, z_fric, J_T, D_T, DimensionTrait<Dim>());
}

//explicit instantiation
template class RigidBodyArchive<float, 2>;
template class RigidBodyArchive<double, 2>;
template class RigidBodyArchive<float, 3>;
template class RigidBodyArchive<double, 3>;

template class RigidBodyDriver<float, 2>;
template class RigidBodyDriver<double, 2>;
template class RigidBodyDriver<float, 3>;
template class RigidBodyDriver<double, 3>;

}
