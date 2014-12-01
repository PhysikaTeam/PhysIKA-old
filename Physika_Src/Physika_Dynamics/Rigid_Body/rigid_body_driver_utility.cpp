/*
 * @file rigid_body_driver_utility.cpp 
 * @The utility class of RigidBodyDriver, providing overload versions of functions for 2D and 3D situations
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
#include "Physika_Dynamics/Rigid_Body/rigid_body_2d.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_3d.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver_utility.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Matrices/sparse_matrix_iterator.h"
#include "Physika_Dynamics/Rigid_Body/rotation_dof.h"
#include "Physika_Dynamics/Rigid_Body/compressed_matrix.h"

#include "Physika_Core/Timer/timer.h"

namespace Physika{

///////////////////////////////////////////////////////////////////////////////////////
//RigidBodyDriverUtility
///////////////////////////////////////////////////////////////////////////////////////


template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::computeMassMatrix(RigidBodyDriver<Scalar, Dim>* driver, CompressedInertiaMatrix<Scalar, Dim>& M, CompressedInertiaMatrix<Scalar, Dim>& M_inv)
{
    RigidBodyDriverUtilityTrait<Scalar>::computeMassMatrix(driver, M, M_inv, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::computeJacobianMatrix(RigidBodyDriver<Scalar, Dim>* driver, CompressedJacobianMatrix<Scalar, Dim>& J)
{
    RigidBodyDriverUtilityTrait<Scalar>::computeJacobianMatrix(driver, J, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::computeFricJacobianMatrix(RigidBodyDriver<Scalar, Dim>* driver, CompressedJacobianMatrix<Scalar, Dim>& D)
{
    RigidBodyDriverUtilityTrait<Scalar>::computeFricJacobianMatrix(driver, D, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::computeGeneralizedVelocity(RigidBodyDriver<Scalar, Dim>* driver, VectorND<Scalar>& v)
{
    RigidBodyDriverUtilityTrait<Scalar>::computeGeneralizedVelocity(driver, v, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::computeCoefficient(RigidBodyDriver<Scalar, Dim>* driver, VectorND<Scalar>& CoR, VectorND<Scalar>& CoF)
{
    RigidBodyDriverUtilityTrait<Scalar>::computeCoefficient(driver, CoR, CoF, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::solveBLCPWithPGS(RigidBodyDriver<Scalar, Dim>* driver, CompressedJacobianMatrix<Scalar, Dim>& J, CompressedJacobianMatrix<Scalar, Dim>& D, CompressedJacobianMatrix<Scalar, Dim>& MJ, CompressedJacobianMatrix<Scalar, Dim>& MD,
    VectorND<Scalar>& pre_Jv, VectorND<Scalar>& post_Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
    VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count)
{
    RigidBodyDriverUtilityTrait<Scalar>::solveBLCPWithPGS(driver, J, D, MJ, MD, pre_Jv, post_Jv, Dv, z_norm, z_fric, CoR, CoF, iteration_count, DimensionTrait<Dim>());
}

template <typename Scalar,int Dim>
void RigidBodyDriverUtility<Scalar, Dim>::applyImpulse(RigidBodyDriver<Scalar, Dim>* driver, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric, CompressedJacobianMatrix<Scalar, Dim>& J_T, CompressedJacobianMatrix<Scalar, Dim>& D_T)
{
    RigidBodyDriverUtilityTrait<Scalar>::applyImpulse(driver, z_norm, z_fric, J_T, D_T, DimensionTrait<Dim>());
}


///////////////////////////////////////////////////////////////////////////////////////
//RigidBodyDriverUtilityTrait
///////////////////////////////////////////////////////////////////////////////////////


template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeMassMatrix(RigidBodyDriver<Scalar, 2>* driver, CompressedInertiaMatrix<Scalar, 2>& M, CompressedInertiaMatrix<Scalar, 2>& M_inv, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeMassMatrix(RigidBodyDriver<Scalar, 3>* driver, CompressedInertiaMatrix<Scalar, 3>& M, CompressedInertiaMatrix<Scalar, 3>& M_inv, DimensionTrait<3> trait)
{
    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }

    //initialize
    unsigned int n = driver->numRigidBody();

    //basic check of matrix's dimensions
    if(M.numElement() != n)
    {
        std::cerr<<"Dimension of matrix M is wrong!"<<std::endl;
        return;
    }
    if(M_inv.numElement() != n)
    {
        std::cerr<<"Dimension of matrix M_inv is wrong!"<<std::endl;
        return;
    }

    //update M and M_inv
    RigidBody<Scalar, 3>* rigid_body = NULL;
    for(unsigned int i = 0; i < n; ++i)
    {
        rigid_body = driver->rigidBody(i);
        if(rigid_body == NULL)
        {
            std::cerr<<"Null rigid body in updating matrix!"<<std::endl;
            continue;
        }
        if(!rigid_body->isFixed())//not fixed
        {
            //mass
            M.setMass(i, rigid_body->mass());
            M_inv.setMass(i, (Scalar)1 / rigid_body->mass());

            //inversed inertia tensor
            M.setInertiaTensor(i, rigid_body->spatialInertiaTensor());
            M_inv.setInertiaTensor(i, rigid_body->spatialInertiaTensorInverse());
        }
        else//fixed
        {
            //mass
            M.setMass(i, 0);
            M_inv.setMass(i, 0);

            //inversed inertia tensor
            M.setInertiaTensor(i, rigid_body->spatialInertiaTensor() * 0);
            M_inv.setInertiaTensor(i, rigid_body->spatialInertiaTensorInverse() * 0);
            //do nothing because all entries are zero
        }
    }
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeJacobianMatrix(RigidBodyDriver<Scalar, 2>* driver, CompressedJacobianMatrix<Scalar, 2>& J, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeJacobianMatrix(RigidBodyDriver<Scalar, 3>* driver, CompressedJacobianMatrix<Scalar, 3>& J, DimensionTrait<3> trait)
{
    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }

    //initialize
    unsigned int m = driver->numContactPoint();
    unsigned int n = driver->numRigidBody();

    //basic check of matrix's dimensions
    if(J.rows() != m || J.cols() != n)
    {
        std::cerr<<"Dimension of matrix J is wrong!"<<std::endl;
        return;
    }

    //update J
    RigidBody<Scalar, 3>* rigid_body = NULL;
    ContactPoint<Scalar, 3>* contact_point = NULL;
    unsigned int object_lhs, object_rhs;
    Vector<Scalar, 3> angular_normal_lhs, angular_normal_rhs;
    for(unsigned int i = 0; i < m; ++i)
    {
        contact_point = driver->contactPoint(i);
        if(contact_point == NULL)
        {
            std::cerr<<"Null contact point in updating matrix!"<<std::endl;
            continue;
        }
        object_lhs = contact_point->objectLhsIndex();
        object_rhs = contact_point->objectRhsIndex();
        rigid_body = driver->rigidBody(object_lhs);
        angular_normal_lhs = (contact_point->globalContactPosition() - rigid_body->globalTranslation()).cross(contact_point->globalContactNormalLhs());
        rigid_body = driver->rigidBody(object_rhs);
        angular_normal_rhs = (contact_point->globalContactPosition() - rigid_body->globalTranslation()).cross(contact_point->globalContactNormalRhs());
        J.setFirstValue(i, object_lhs, -contact_point->globalContactNormalLhs(), -angular_normal_lhs);
        J.setSecondValue(i, object_rhs, -contact_point->globalContactNormalRhs(), -angular_normal_rhs);
    }
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeFricJacobianMatrix(RigidBodyDriver<Scalar, 2>* driver, CompressedJacobianMatrix<Scalar, 2>& D, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeFricJacobianMatrix(RigidBodyDriver<Scalar, 3>* driver, CompressedJacobianMatrix<Scalar, 3>& D, DimensionTrait<3> trait)
{
    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }

    //initialize
    unsigned int m = driver->numContactPoint();
    unsigned int n = driver->numRigidBody();
    unsigned int s = D.rows();
    unsigned int fric_sample_count = s / m;

    //basic check of matrix's dimensions
    if(D.rows() != s || D.cols() != n || s != m * fric_sample_count)
    {
        std::cerr<<"Dimension of matrix D is wrong!"<<std::endl;
        return;
    }

    //update D
    RigidBody<Scalar, 3>* rigid_body = NULL;
    ContactPoint<Scalar, 3>* contact_point = NULL;
    unsigned int object_lhs, object_rhs;
    Vector<Scalar, 3> angular_normal_lhs, angular_normal_rhs;
    for(unsigned int i = 0; i < m; ++i)
    {
        contact_point = driver->contactPoint(i);
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
            rigid_body = driver->rigidBody(object_lhs);
            angular_normal_lhs = (contact_point->globalContactPosition() - rigid_body->globalTranslation()).cross(sample_normal);
            rigid_body = driver->rigidBody(object_rhs);
            angular_normal_rhs = (contact_point->globalContactPosition() - rigid_body->globalTranslation()).cross(sample_normal);
            D.setFirstValue(i * fric_sample_count + k, object_lhs, sample_normal, angular_normal_lhs);
            D.setSecondValue(i * fric_sample_count + k, object_rhs, sample_normal, angular_normal_rhs);
            sample_normal = rotation.rotate(sample_normal);
        }
    }
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeGeneralizedVelocity(RigidBodyDriver<Scalar, 2>* driver, VectorND<Scalar>& v, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeGeneralizedVelocity(RigidBodyDriver<Scalar, 3>* driver, VectorND<Scalar>& v, DimensionTrait<3> trait)
{
    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }

    //initialize
    unsigned int n = driver->numRigidBody();
    unsigned int six_n = n * 6;

    //basic check of matrix's dimensions
    if(v.dims() != six_n)
    {
        std::cerr<<"Dimension of vector v is wrong!"<<std::endl;
        return;
    }

    //update v
    RigidBody<Scalar, 3>* rigid_body = NULL;
    for(unsigned int i = 0; i < n; ++i)
    {
        rigid_body = driver->rigidBody(i);
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

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeCoefficient(RigidBodyDriver<Scalar, 2>* driver, VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::computeCoefficient(RigidBodyDriver<Scalar, 3>* driver, VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, DimensionTrait<3> trait)
{
    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }

    //initialize
    unsigned int m = driver->numContactPoint();
    unsigned int s = CoF.dims();
    unsigned int fric_sample_count = s / m;

    //dimension check
    if(CoR.dims() != m || m * fric_sample_count != s)
    {
        std::cerr<<"Wrong dimension in updating coefficient!"<<std::endl;
        return;
    }

    //update CoR and CoF
    ContactPoint<Scalar, 3>* contact_point = NULL;
    RigidBody<Scalar, 3>* rigid_body_lhs, * rigid_body_rhs;
    Scalar cor_lhs, cor_rhs, cof_lhs, cof_rhs;
    for(unsigned int i = 0; i < m; ++i)
    {
        contact_point = driver->contactPoint(i);
        if(contact_point == NULL)
        {
            std::cerr<<"Null contact point in updating coefficient!"<<std::endl;
            continue;
        }
        rigid_body_lhs = driver->rigidBody(contact_point->objectLhsIndex());
        rigid_body_rhs = driver->rigidBody(contact_point->objectRhsIndex());
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

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::solveBLCPWithPGS(RigidBodyDriver<Scalar, 2>* driver, CompressedJacobianMatrix<Scalar, 2>& J, CompressedJacobianMatrix<Scalar, 2>& D, CompressedJacobianMatrix<Scalar, 2>& MJ, CompressedJacobianMatrix<Scalar, 2>& MD,
    VectorND<Scalar>& pre_Jv, VectorND<Scalar>& post_Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
    VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count,
    DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::solveBLCPWithPGS(RigidBodyDriver<Scalar, 3>* driver, CompressedJacobianMatrix<Scalar, 3>& J, CompressedJacobianMatrix<Scalar, 3>& D, CompressedJacobianMatrix<Scalar, 3>& MJ, CompressedJacobianMatrix<Scalar, 3>& MD,
    VectorND<Scalar>& pre_Jv, VectorND<Scalar>& post_Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
    VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count,
    DimensionTrait<3> trait)
{
    //dimension check is temporary ignored because its too long to write here
    //a function to perform such check will be introduced 

    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }
    //initialize
    unsigned int m = driver->numContactPoint();
    unsigned int n = driver->numRigidBody();
    unsigned int s = D.rows();
    unsigned int fric_sample_count = s / m;
    z_norm = -1 * pre_Jv;
    z_fric *= 0;
    VectorND<Scalar> MJz_norm_temp = MJ * z_norm;
    VectorND<Scalar> MDz_fric_temp = MD * z_fric;
    std::vector<VectorND<Scalar> > MJz_norm, MDz_fric;
    VectorND<Scalar> element(6);
    for(unsigned int i = 0; i < n; ++i)
    {
        for(unsigned int j = 0; j < 6; ++j)
        {
            element[j] = MJz_norm_temp[i * 6 + j];
        }
        MJz_norm.push_back(element);
    }
    for(unsigned int i = 0; i < n; ++i)
    {
        for(unsigned int j = 0; j < 6; ++j)
        {
            element[j] = MDz_fric_temp[i * 6 + j];
        }
        MDz_fric.push_back(element);
    }
    
    Scalar delta, m_value;
    //iteration
    for(unsigned int itr = 0; itr < iteration_count; ++itr)
    {
        //normal contact step
        for(unsigned int i = 0; i < m; ++i)
        {
            unsigned int object_lhs, object_rhs;
            object_lhs = J.firstColumnIndex(i);
            object_rhs = J.secondColumnIndex(i);

            m_value = productValue(J, MJ, i, i);
            
            if(m_value != 0)
            {
                Scalar JMDz_fric = J.firstValue(i).dot(MDz_fric[object_lhs]) + J.secondValue(i).dot(MDz_fric[object_rhs]);
                delta = J.firstValue(i).dot(MJz_norm[object_lhs]) + J.secondValue(i).dot(MJz_norm[object_rhs]);
                delta = (post_Jv[i] - pre_Jv[i] - JMDz_fric - delta) / m_value;
            }
            else
                delta = 0;
            
            Scalar z_normal_origin = z_norm[i];
            z_norm[i] += delta;
            if(z_norm[i] < 0)
                z_norm[i] = 0;
            delta = z_norm[i] - z_normal_origin;
            MJz_norm[object_lhs] += delta * MJ.firstValue(i);
            MJz_norm[object_rhs] += delta * MJ.secondValue(i);
        }
        //friction step
        for(unsigned int i = 0; i < s; ++i)
        {
            unsigned int object_lhs, object_rhs;
            object_lhs = D.firstColumnIndex(i);
            object_rhs = D.secondColumnIndex(i);

            m_value = productValue(D, MD, i, i);

            if(m_value != 0)
            {
                Scalar DMJz_norm = D.firstValue(i).dot(MJz_norm[object_lhs]) + D.secondValue(i).dot(MJz_norm[object_rhs]);
                delta = D.firstValue(i).dot(MDz_fric[object_lhs]) + D.secondValue(i).dot(MDz_fric[object_rhs]);
                delta = (-Dv[i] - DMJz_norm - delta) / m_value;
            }
            else
                delta = 0;

            Scalar z_fric_origin = z_fric[i];
            z_fric[i] += delta;
            if(z_fric[i] < - CoF[i / fric_sample_count] * z_norm[i / fric_sample_count])
                z_fric[i] = - CoF[i / fric_sample_count] * z_norm[i / fric_sample_count];
            if(z_fric[i] > CoF[i / fric_sample_count] * z_norm[i / fric_sample_count])
                z_fric[i] = CoF[i / fric_sample_count] * z_norm[i / fric_sample_count];
            delta = z_fric[i] - z_fric_origin;
            MDz_fric[object_lhs] += delta * MD.firstValue(i);
            MDz_fric[object_rhs] += delta * MD.secondValue(i);
        }
    }
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::applyImpulse(RigidBodyDriver<Scalar, 2>* driver, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric, 
    CompressedJacobianMatrix<Scalar, 2>& J_T, CompressedJacobianMatrix<Scalar, 2>& D_T, DimensionTrait<2> trait)
{
    //to do
}

template <typename Scalar>
void RigidBodyDriverUtilityTrait<Scalar>::applyImpulse(RigidBodyDriver<Scalar, 3>* driver, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
    CompressedJacobianMatrix<Scalar, 3>& J_T, CompressedJacobianMatrix<Scalar, 3>& D_T, DimensionTrait<3> trait)
{
    if(driver == NULL)
    {
        std::cerr<<"Null driver!"<<std::endl;
        return;
    }

    //initialize
    unsigned int m = driver->numContactPoint();
    unsigned int n = driver->numRigidBody();
    unsigned int s = D_T.rows();
    unsigned int fric_sample_count = s / m;

    //basic check of matrix's dimensions
    if(J_T.rows() != m || J_T.cols() != n)
    {
        std::cerr<<"Dimension of matrix J_T is wrong!"<<std::endl;
        return;
    }
    if(D_T.rows() != s || D_T.cols() != n)
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
    VectorND<Scalar> impulse_translation = J_T * z_norm;
    VectorND<Scalar> impulse_angular = D_T * z_fric;

    //apply impulses to rigid bodies. This step will not cause velocity and configuration integral 
    RigidBody<Scalar, 3>* rigid_body = NULL;
    for(unsigned int i = 0; i < n; ++i)
    {
        rigid_body = driver->rigidBody(i);
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

template class RigidBodyDriverUtility<float, 2>;
template class RigidBodyDriverUtility<double, 2>;
template class RigidBodyDriverUtility<float, 3>;
template class RigidBodyDriverUtility<double, 3>;

template class RigidBodyDriverUtilityTrait<float>;
template class RigidBodyDriverUtilityTrait<double>;

}