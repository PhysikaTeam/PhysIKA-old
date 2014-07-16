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
	if(rigid_body == NULL)
		return;

	rigid_body_ = rigid_body;

	switch(rigid_body->objectType())
	{
		case CollidableObjectInternal::MESH_BASED: collide_object_ = new MeshBasedCollidableObject<Scalar>();break;
		default: std::cerr<<"Object type error!"<<std::endl; return;
	}
	MeshBasedCollidableObject<Scalar>* mesh_object = dynamic_cast<MeshBasedCollidableObject<Scalar>*>(collide_object_);
	mesh_object->setMesh(rigid_body->mesh());
	mesh_object->setTransform(rigid_body->transformPtr());

	object_bvh_ = new ObjectBVH<Scalar, Dim>();
	object_bvh_->setCollidableObject(collide_object_);

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
void RigidBodyDriver<Scalar, Dim>::initialize()
{
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
void RigidBodyDriver<Scalar, Dim>::write(const std::string &file_name)
{
    //plugin
	unsigned int plugin_num = static_cast<unsigned int>((this->plugins_).size());
	RigidDriverPlugin<Scalar, Dim>* plugin;
	for(unsigned int i = 0; i < plugin_num; ++i)
	{
		plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
		if(plugin != NULL)
			plugin->onWrite(frame_);
	}
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::read(const std::string &file_name)
{
    //plugin
	unsigned int plugin_num = static_cast<unsigned int>((this->plugins_).size());
	RigidDriverPlugin<Scalar, Dim>* plugin;
	for(unsigned int i = 0; i < plugin_num; ++i)
	{
		plugin = dynamic_cast<RigidDriverPlugin<Scalar, Dim>*>((this->plugins_)[i]);
		if(plugin != NULL)
			plugin->onRead(frame_);
	}
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

    //update the matrix of dynamics
    updateDynamicsMatrix(J, M_inv, D, v);

    //calculate other matrix in need
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

    // update CoR and CoF
    updateCoefficient(CoR, CoF);

    //solve BLCP with PGS. z_norm and z_fric are the unknown variables
    solveBLCPWithPGS(JMJ, DMD, JMD, DMJ, Jv, Dv, z_norm, z_fric, CoR, CoF);

    //apply impulse
    applyImpulse(z_norm, z_fric, J_T, D_T);
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::updateDynamicsMatrix(SparseMatrix<Scalar>& J, SparseMatrix<Scalar>& M_inv, SparseMatrix<Scalar>& D, VectorND<Scalar>& v)
{
    //initialize
    unsigned int m = contact_points_.numContactPoint();
    unsigned int n = numRigidBody();
    unsigned int six_n = n * 6;
    unsigned int s = D.rows();
    unsigned int fric_sample_count = s / m;

    //basic check of matrix's dimensions
    if(J.rows() != m || J.cols() != six_n)
    {
        std::cerr<<"Dimension of matrix J is wrong!"<<std::endl;
        return;
    }
    if(M_inv.rows() != six_n || M_inv.cols() != six_n)
    {
        std::cerr<<"Dimension of matrix M_inv is wrong!"<<std::endl;
        return;
    }
    if(D.rows() != s || D.cols() != six_n || s != m * fric_sample_count)
    {
        std::cerr<<"Dimension of matrix D is wrong!"<<std::endl;
        return;
    }
    if(v.dims() != six_n)
    {
        std::cerr<<"Dimension of vector v is wrong!"<<std::endl;
        return;
    }

    //update M_inv
    RigidBody<Scalar, Dim>* rigid_body = NULL;
    for(unsigned int i = 0; i < n; ++i)
    {
        rigid_body = rigidBody(i);
        if(rigid_body == NULL)
        {
            std::cerr<<"Null rigid body in updating matrix!"<<std::endl;
            continue;
        }
        if(!rigid_body->isFixed())//not fixed
        {
            //inversed mass
            M_inv.setEntry(6 * i, 6 * i, (Scalar)1 / rigid_body->mass());
            M_inv.setEntry(6 * i + 1, 6 * i + 1, (Scalar)1 / rigid_body->mass());
            M_inv.setEntry(6 * i + 2, 6 * i + 2, (Scalar)1 / rigid_body->mass());
            //inversed inertia tensor
            for(unsigned int j = 0; j < 3; ++j)
            {
                for(unsigned int k = 0; k < 3; ++k)
                {
                    M_inv.setEntry(6 * i + 3 + j, 6 * i + 3 + k, rigid_body->spatialInertiaTensorInverse()(j, k));
                }
            }
        }
        else//fixed
        {
            //do nothing because all entries are zero
        }
    }

    //update J
    ContactPoint<Scalar, Dim>* contact_point = NULL;
    unsigned int object_lhs, object_rhs;
    Vector<Scalar, 3> angular_normal_lhs, angular_normal_rhs;
    for(unsigned int i = 0; i < m; ++i)
    {
        contact_point = contact_points_[i];
        if(contact_point == NULL)
        {
            std::cerr<<"Null contact point in updating matrix!"<<std::endl;
            continue;
        }
        object_lhs = contact_point->objectLhsIndex();
        object_rhs = contact_point->objectRhsIndex();
        angular_normal_lhs = (contact_point->globalContactPosition() - rigidBody(object_lhs)->globalTranslation()).cross(contact_point->globalContactNormalLhs());
        angular_normal_rhs = (contact_point->globalContactPosition() - rigidBody(object_rhs)->globalTranslation()).cross(contact_point->globalContactNormalRhs());
        for(unsigned int j = 0; j < 3; ++j)
        {
            J.setEntry(i, object_lhs * 6 + j, contact_point->globalContactNormalLhs()[j]);
            J.setEntry(i, object_rhs * 6 + j, contact_point->globalContactNormalRhs()[j]);
        }
        for(unsigned int j = 0; j < 3; ++j)
        {
            J.setEntry(i, object_lhs * 6 + 3 + j, angular_normal_lhs[j]);
            J.setEntry(i, object_rhs * 6 + 3 + j, angular_normal_rhs[j]);
        }
    }

    //update D
    for(unsigned int i = 0; i < m; ++i)
    {
        contact_point = contact_points_[i];
        if(contact_point == NULL)
        {
            std::cerr<<"Null contact point in updating matrix!"<<std::endl;
            continue;
        }
        object_lhs = contact_point->objectLhsIndex();
        object_rhs = contact_point->objectRhsIndex();

        //friction direction sampling: rotate around the normal for fric_sample_count times
        Vector<Scalar, 3> contact_normal_lhs = contact_point->globalContactNormalLhs();
        Quaternion<Scalar> rotation(contact_normal_lhs, PI / fric_sample_count);
        Vector<Scalar, 3> sample_normal;

        //step one, find an arbitrary unit vector orthogonal to contact normal
        if(contact_normal_lhs[0] <= std::numeric_limits<Scalar>::epsilon() && contact_normal_lhs[1] <= std::numeric_limits<Scalar>::epsilon())//(0, 0, 1)
        {
            sample_normal = Vector<Scalar, 3>(1, 0, 0);
        }
        else
        {
            sample_normal = contact_normal_lhs.cross(Vector<Scalar, 3>(0, 0, 1));
        }

        //step two, rotate around the normal for fric_sample_count times to get fric_sample_count normal samples
        for(unsigned int k = 0; k< fric_sample_count; ++k)
        {
            sample_normal.normalize();
            angular_normal_lhs = (contact_point->globalContactPosition() - rigidBody(object_lhs)->globalTranslation()).cross(sample_normal);
            angular_normal_rhs = (contact_point->globalContactPosition() - rigidBody(object_rhs)->globalTranslation()).cross(sample_normal);
            for(unsigned int j = 0; j < 3; ++j)
            {
                D.setEntry(i * fric_sample_count + k, object_lhs * 6 + j, sample_normal[j]);
                D.setEntry(i * fric_sample_count + k, object_rhs * 6 + j, sample_normal[j]);
            }
            for(unsigned int j = 0; j < 3; ++j)
            {
                D.setEntry(i * fric_sample_count + k, object_lhs * 6 + 3 + j, angular_normal_lhs[j]);
                D.setEntry(i * fric_sample_count + k, object_rhs * 6 + 3 + j, angular_normal_rhs[j]);
            }
            sample_normal = rotation.rotate(sample_normal);
        }
    }

    //update v
    for(unsigned int i = 0; i < n; ++i)
    {
        rigid_body = rigidBody(i);
        if(rigid_body == NULL)
        {
            std::cerr<<"Null rigid body in updating matrix!"<<std::endl;
            continue;
        }
        for(unsigned int j = 0; j < 3; ++j)
        {
            v[6 * i + j] = rigid_body->globalTranslationVelocity()[j];
            v[6 * i + 3 + j] = rigid_body->globalAngularVelocity()[j];
        }
    }

}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::updateCoefficient(VectorND<Scalar>& CoR, VectorND<Scalar>& CoF)
{
    //initialize
    unsigned int m = contact_points_.numContactPoint();
    unsigned int s = CoF.dims();
    unsigned int fric_sample_count = s / m;

    //dimension check
    if(CoR.dims() != m || m * fric_sample_count != s)
    {
        std::cerr<<"Wrong dimension in updating coefficient!"<<std::endl;
        return;
    }

    //update CoR and CoF
    ContactPoint<Scalar, Dim>* contact_point = NULL;
    RigidBody<Scalar, Dim>* rigid_body_lhs, * rigid_body_rhs;
    Scalar cor_lhs, cor_rhs, cof_lhs, cof_rhs;
    for(unsigned int i = 0; i < m; ++i)
    {
        contact_point = contact_points_[i];
        if(contact_point == NULL)
        {
            std::cerr<<"Null contact point in updating coefficient!"<<std::endl;
            continue;
        }
        rigid_body_lhs = rigidBody(contact_point->objectLhsIndex());
        rigid_body_rhs = rigidBody(contact_point->objectRhsIndex());
        if(rigid_body_lhs == NULL || rigid_body_rhs == NULL)
        {
            std::cerr<<"Null rigid body in updating coefficient!"<<std::endl;
            continue;
        }
        cor_lhs = rigid_body_lhs->coeffRestitution();
        cor_rhs = rigid_body_rhs->coeffRestitution();
        cof_lhs = rigid_body_lhs->coeffFriction();
        cof_rhs = rigid_body_rhs->coeffFriction();
        CoR[i] = min(cor_lhs, cor_rhs);
        for(unsigned int j = 0; j< fric_sample_count; ++j)
        {
            CoF[i * fric_sample_count + j] = max(cof_lhs, cof_rhs);
        }
    }
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::solveBLCPWithPGS(SparseMatrix<Scalar>& JMJ, SparseMatrix<Scalar>& DMD, SparseMatrix<Scalar>& JMD, SparseMatrix<Scalar>& DMJ,
                                                    VectorND<Scalar>& Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
                                                    VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count)
{
    //dimension check is temporary ignored because its too long to write here
    //a function to perform such check will be introduced 
    
    //initialize
    unsigned int m = contact_points_.numContactPoint();
    unsigned int n = numRigidBody();
    unsigned int six_n = n * 6;
    unsigned int s = DMD.cols();
    unsigned int fric_sample_count = s / m;
    z_norm = Jv;
    z_fric *= 0;
    VectorND<Scalar> JMDz_fric;
    VectorND<Scalar> DMJz_norm;
    std::vector<Trituple<Scalar>> non_zeros;
    Scalar delta, m_value;

    //iteration
    for(unsigned int itr = 0; itr < iteration_count; ++itr)
    {
        //normal contact step
        JMDz_fric = JMD * z_fric;
        for(unsigned int i = 0; i < m; ++i)
        {
            delta = 0;
            non_zeros.clear();
            non_zeros = JMJ.getRowElements(i);
            unsigned int size_non_zeros = static_cast<unsigned int>(non_zeros.size());
            for(unsigned int j = 0; j < size_non_zeros; ++j)
            {
                if(non_zeros[j].col_ != i)//not diag
                {
                    delta += non_zeros[j].value_ * z_norm[non_zeros[j].col_];
                }
            }
            m_value = JMJ(i, i);

            if(m_value != 0)
                delta = (Jv[i] - JMDz_fric[i] - delta) / m_value;
            else
                delta = 0;
            
            if(delta < 0)
                z_norm[i] = 0;
            else
                z_norm[i] = delta;
        }
        //friction step
        DMJz_norm = DMJ * z_norm;
        for(unsigned int i = 0; i < s; ++i)
        {
            delta = 0;
            non_zeros.clear();
            non_zeros = DMD.getRowElements(i);
            unsigned int size_non_zeros = static_cast<unsigned int>(non_zeros.size());
            for(unsigned int j = 0; j < size_non_zeros; ++j)
            {
                if(non_zeros[j].col_ != i)//not diag
                    delta += non_zeros[j].value_ * z_fric[non_zeros[j].col_];
            }
            m_value = DMD(i, i);
            if(m_value != 0)
                delta = (Dv[i] - DMJz_norm[i] - delta) / m_value;
            else
                delta = 0;
            if(delta < - CoF[i / fric_sample_count] * z_norm[i / fric_sample_count])
                z_fric[i] = - CoF[i / fric_sample_count] * z_norm[i / fric_sample_count];
            if(delta > CoF[i / fric_sample_count] * z_norm[i / fric_sample_count])
                z_fric[i] = CoF[i / fric_sample_count] * z_norm[i / fric_sample_count];
        }
    }
}

template <typename Scalar,int Dim>
void RigidBodyDriver<Scalar, Dim>::applyImpulse(VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric, SparseMatrix<Scalar>& J_T, SparseMatrix<Scalar>& D_T)
{
    //initialize
    unsigned int m = contact_points_.numContactPoint();
    unsigned int n = numRigidBody();
    unsigned int six_n = n * 6;
    unsigned int s = D_T.cols();
    unsigned int fric_sample_count = s / m;

    //basic check of matrix's dimensions
    if(J_T.rows() != six_n || J_T.cols() != m)
    {
        std::cerr<<"Dimension of matrix J_T is wrong!"<<std::endl;
        return;
    }
    if(D_T.rows() != six_n || D_T.cols() != s)
    {
        std::cerr<<"Dimension of matrix D_T is wrong!"<<std::endl;
        return;
    }
    if(z_norm.dims() != m)
    {
        std::cerr<<"Dimension of matrix z_norm is wrong!"<<std::endl;
        return;
    }
    if(z_fric.dims() != s)
    {
        std::cerr<<"Dimension of matrix z_fric is wrong!"<<std::endl;
        return;
    }

    //calculate impulses from their magnitudes (z_norm, z_fric) and directions (J_T, D_T)
    VectorND<Scalar> impulse_translation = J_T * z_norm * (-1);
    VectorND<Scalar> impulse_angular = D_T * z_fric * (-1);

    //apply impulses to rigid bodies. This step will not cause velocity and configuration integral 
    RigidBody<Scalar, Dim>* rigid_body = NULL;
    for(unsigned int i = 0; i < n; ++i)
    {
        rigid_body = rigidBody(i);
        if(rigid_body == NULL)
        {
            std::cerr<<"Null rigid body in updating matrix!"<<std::endl;
            continue;
        }
        VectorND<Scalar> impulse(6, 0);
        for(unsigned int j = 0; j < 6; ++j)
        {
            impulse[j] = impulse_translation[6 * i + j] + impulse_angular[6 * i + j];
        }
        rigid_body->addTranslationImpulse(Vector<Scalar, 3>(impulse[0], impulse[1], impulse[2]));
        rigid_body->addAngularImpulse(Vector<Scalar, 3>(impulse[3], impulse[4], impulse[5]));
    }
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

//explicit instantiation
template class RigidBodyArchive<float, 3>;
template class RigidBodyArchive<double, 3>;
template class RigidBodyDriver<float, 3>;
template class RigidBodyDriver<double, 3>;


}
