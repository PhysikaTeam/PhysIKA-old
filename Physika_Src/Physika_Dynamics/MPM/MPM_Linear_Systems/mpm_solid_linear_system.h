/*
 * @file mpm_solid_linear_system.h
 * @brief linear system for implicit integration of MPMSolid && CPDIMPMSolid driver
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_MPM_SOLID_LINEAR_SYSTEM_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_MPM_SOLID_LINEAR_SYSTEM_H_

#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"
#include "Physika_Dynamics/Utilities/Grid_Generalized_Vectors/uniform_grid_generalized_vector_TV.h"

namespace Physika{

template <typename Scalar, int Dim> class MPMSolid;

/*
 * MPMSolidLinearSystem: linear system for implicit integration of MPMSolid driver
 * x and b are represented by UniformGridGeneralizedVector
 */

template <typename Scalar, int Dim>
class MPMSolidLinearSystem: public LinearSystem<Scalar>
{
public:
    explicit MPMSolidLinearSystem(MPMSolid<Scalar,Dim> *mpm_solid_driver);
    virtual ~MPMSolidLinearSystem();
    virtual void multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
    virtual void preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
    virtual Scalar innerProduct(const GeneralizedVector<Scalar> &x, const GeneralizedVector<Scalar> &y) const;
    //set the object to construct the linear system for (each object of MPMSolid is solved independently)
    //construct one global linear system for all objects if obj_idx is set to -1 (all objects are solved on one grid)
    void setActiveObject(int obj_idx);
protected:
    //potential energy Hessian acted on an arbitrary increment x_diff
    void energyHessianMultiply(const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &x_diff,
                               UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &result) const;
    //diagonal of energy Hessian acted on an arbitrary increment x_diff, for jacobi precondition
    void energyHessianDiagonalMultiply(const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &x_diff,
                                       UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &result) const;
    void jacobiPreconditionerMultiply(const UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &x,
                                      UniformGridGeneralizedVector<Vector<Scalar, Dim>, Dim> &result) const;
    //disable default copy
    MPMSolidLinearSystem();
    MPMSolidLinearSystem(const MPMSolidLinearSystem<Scalar,Dim> &linear_system);
    MPMSolidLinearSystem<Scalar,Dim>& operator= (const MPMSolidLinearSystem<Scalar,Dim> &linear_system);
protected:
    MPMSolid<Scalar,Dim> *mpm_solid_driver_;
    int active_obj_idx_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_MPM_SOLID_LINEAR_SYSTEM_H_
