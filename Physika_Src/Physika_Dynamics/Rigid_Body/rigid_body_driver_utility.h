/*
 * @file rigid_body_driver_utility.h 
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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_UTILITY_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_UTILITY_H_

#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"

namespace Physika{

template <typename Scalar>
class RigidBodyDriverUtility
{
public:
    //overload versions of utilities for 2D and 3D situations
    static void computeInvMassMatrix(RigidBodyDriver<Scalar, 2>* driver, SparseMatrix<Scalar>& M_inv, DimensionTrait<2> trait);
    static void computeInvMassMatrix(RigidBodyDriver<Scalar, 3>* driver, SparseMatrix<Scalar>& M_inv, DimensionTrait<3> trait);

    static void computeJacobianMatrix(RigidBodyDriver<Scalar, 2>* driver, SparseMatrix<Scalar>& J, DimensionTrait<2> trait);
    static void computeJacobianMatrix(RigidBodyDriver<Scalar, 3>* driver, SparseMatrix<Scalar>& J, DimensionTrait<3> trait);

    static void computeFricJacobianMatrix(RigidBodyDriver<Scalar, 2>* driver, SparseMatrix<Scalar>& D, DimensionTrait<2> trait);
    static void computeFricJacobianMatrix(RigidBodyDriver<Scalar, 3>* driver, SparseMatrix<Scalar>& D, DimensionTrait<3> trait);

    static void computeGeneralizedVelocity(RigidBodyDriver<Scalar, 2>* driver, VectorND<Scalar>& v, DimensionTrait<2> trait);
    static void computeGeneralizedVelocity(RigidBodyDriver<Scalar, 3>* driver, VectorND<Scalar>& v, DimensionTrait<3> trait);

    static void computeCoefficient(RigidBodyDriver<Scalar, 2>* driver, VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, DimensionTrait<2> trait);
    static void computeCoefficient(RigidBodyDriver<Scalar, 3>* driver, VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, DimensionTrait<3> trait);

    static void solveBLCPWithPGS(RigidBodyDriver<Scalar, 2>* driver, SparseMatrix<Scalar>& JMJ, SparseMatrix<Scalar>& DMD, SparseMatrix<Scalar>& JMD, SparseMatrix<Scalar>& DMJ,
                                 VectorND<Scalar>& Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
                                 VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count,
                                 DimensionTrait<2> trait);
    static void solveBLCPWithPGS(RigidBodyDriver<Scalar, 3>* driver, SparseMatrix<Scalar>& JMJ, SparseMatrix<Scalar>& DMD, SparseMatrix<Scalar>& JMD, SparseMatrix<Scalar>& DMJ,
                                 VectorND<Scalar>& Jv, VectorND<Scalar>& Dv, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric,
                                 VectorND<Scalar>& CoR, VectorND<Scalar>& CoF, unsigned int iteration_count,
                                 DimensionTrait<3> trait);

    static void applyImpulse(RigidBodyDriver<Scalar, 2>* driver, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric, 
                             SparseMatrix<Scalar>& J_T, SparseMatrix<Scalar>& D_T, DimensionTrait<2> trait);
    static void applyImpulse(RigidBodyDriver<Scalar, 3>* driver, VectorND<Scalar>& z_norm, VectorND<Scalar>& z_fric, 
                             SparseMatrix<Scalar>& J_T, SparseMatrix<Scalar>& D_T, DimensionTrait<3> trait);

};

}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_RIGID_BODY_DRIVER_UTILITY_H_