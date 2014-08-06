/*
 * @file rigid_response_method_BLCP.cpp
 * @Rigid-body collision response using BLCP in
 * "Mass Splitting for Jitter-Free Parallel Rigid Body Simulation"
 * Tonge et al. 2012
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

#include <stdio.h>
#include "Physika_Dynamics/Rigid_Body/rigid_response_method_BLCP.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver_utility.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar, int Dim>
RigidResponseMethodBLCP<Scalar, Dim>::RigidResponseMethodBLCP()
{

}

template <typename Scalar, int Dim>
RigidResponseMethodBLCP<Scalar, Dim>::~RigidResponseMethodBLCP()
{

}

template <typename Scalar, int Dim>
void RigidResponseMethodBLCP<Scalar, Dim>::collisionResponse()
{
    //initialize
    unsigned int m = this->rigid_driver_->numContactPoint();//m: number of contact points
    unsigned int n = this->rigid_driver_->numRigidBody();//n: number of rigid bodies
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
    RigidBodyDriverUtility<Scalar, Dim>::computeInvMassMatrix(this->rigid_driver_, M_inv);
    RigidBodyDriverUtility<Scalar, Dim>::computeJacobianMatrix(this->rigid_driver_, J);
    RigidBodyDriverUtility<Scalar, Dim>::computeFricJacobianMatrix(this->rigid_driver_, D);
    RigidBodyDriverUtility<Scalar, Dim>::computeGeneralizedVelocity(this->rigid_driver_, v);

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
    RigidBodyDriverUtility<Scalar, Dim>::computeCoefficient(this->rigid_driver_, CoR, CoF);

    //solve BLCP with PGS. z_norm and z_fric are the unknown variables
    RigidBodyDriverUtility<Scalar, Dim>::solveBLCPWithPGS(this->rigid_driver_, JMJ, DMD, JMD, DMJ, Jv, Dv, z_norm, z_fric, CoR, CoF);

    //apply impulse
    RigidBodyDriverUtility<Scalar, Dim>::applyImpulse(this->rigid_driver_, z_norm, z_fric, J_T, D_T);
}

template class RigidResponseMethodBLCP<float, 2>;
template class RigidResponseMethodBLCP<double, 2>;
template class RigidResponseMethodBLCP<float, 3>;
template class RigidResponseMethodBLCP<double, 3>;

}
