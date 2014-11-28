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
#include "Physika_Dynamics/Rigid_Body/rotation_dof.h"
#include "Physika_Dynamics/Rigid_Body/compressed_matrix.h"

#include "Physika_Core/Timer/timer.h"

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

    unsigned int dof = n * (Dim + RotationDof<Dim>::degree);//DoF(Degree of Freedom) of a rigid-body system
    unsigned int fric_sample_count = 2;//count of friction sample directions
    unsigned int s = m * fric_sample_count;//s: number of friction sample. Here a square sample is adopted
    CompressedJacobianMatrix<Scalar, Dim> J(m, n);//Jacobian matrix. (m, dof) dimension when uncompressed, (m, 12) dimension after compression
    CompressedJacobianMatrix<Scalar, Dim> D(s, n);//Jacobian matrix of friction. (s, dof) dimension when uncompressed, (s, 12) dimension after compression
    CompressedInertiaMatrix<Scalar, Dim> M(n);//inertia matrix. (dof, dof) dimension when uncompressed, (dof, 6) dimension after compression
    CompressedInertiaMatrix<Scalar, Dim> M_inv(n);//inversed inertia matrix. (dof, dof) dimension when uncompressed, (dof, 6) dimension after compression
    CompressedJacobianMatrix<Scalar, Dim> MJ(n, m);//M * J. (dof, m) dimension when uncompressed, (12, m) dimension after compression
    CompressedJacobianMatrix<Scalar, Dim> MD(n, s);//M * D. (dof, s) dimension when uncompressed, (12, s) dimension after compression
    VectorND<Scalar> v(dof, 0);//generalized velocity of the system
    VectorND<Scalar> Jv(m, 0);//normal relative velocity of each contact point (for normal contact impulse calculation)
    VectorND<Scalar> post_Jv(m, 0);//expected post-impact normal relative velocity of each contact point (for normal contact impulse calculation)
    VectorND<Scalar> Dv(s, 0);//tangent relative velocity of each contact point (for frictional contact impulse calculation)
    VectorND<Scalar> CoR(m, 0);//coefficient of restitution (for normal contact impulse calculation)
    VectorND<Scalar> CoF(s, 0);//coefficient of friction (for frictional contact impulse calculation)
    VectorND<Scalar> z_norm(m, 0);//normal contact impulse. The key of collision response
    VectorND<Scalar> z_fric(s, 0);//frictional contact impulse. The key of collision response

    RigidBodyDriverUtility<Scalar, Dim>::computeMassMatrix(this->rigid_driver_, M, M_inv);
    RigidBodyDriverUtility<Scalar, Dim>::computeJacobianMatrix(this->rigid_driver_, J);
    RigidBodyDriverUtility<Scalar, Dim>::computeFricJacobianMatrix(this->rigid_driver_, D);
    RigidBodyDriverUtility<Scalar, Dim>::computeGeneralizedVelocity(this->rigid_driver_, v);
    
    //compute other matrix in need
    CompressedJacobianMatrix<Scalar, Dim> J_T = J;
    J_T = J.transpose();
    CompressedJacobianMatrix<Scalar, Dim> D_T = D;
    D_T = D.transpose();
    MJ = M_inv * J_T;
    MD = M_inv * D_T;
    Jv = J * v;
    Dv = D * v;

    //update CoR and CoF
    RigidBodyDriverUtility<Scalar, Dim>::computeCoefficient(this->rigid_driver_, CoR, CoF);
    
    //calculate the expected post-impact velocity
    for(unsigned int i = 0; i < m; ++i)
        post_Jv[i] = -Jv[i] * CoR[i];

    //solve BLCP with PGS. z_norm and z_fric are the unknown variables
    RigidBodyDriverUtility<Scalar, Dim>::solveBLCPWithPGS(this->rigid_driver_, J, D, MJ, MD, Jv, post_Jv, Dv, z_norm, z_fric, CoR, CoF, 20);
    //apply impulse
    RigidBodyDriverUtility<Scalar, Dim>::applyImpulse(this->rigid_driver_, z_norm, z_fric, J_T, D_T);
}

template class RigidResponseMethodBLCP<float, 2>;
template class RigidResponseMethodBLCP<double, 2>;
template class RigidResponseMethodBLCP<float, 3>;
template class RigidResponseMethodBLCP<double, 3>;

}
