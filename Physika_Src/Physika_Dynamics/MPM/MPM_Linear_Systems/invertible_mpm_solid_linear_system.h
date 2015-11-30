/*
* @file invertible_mpm_solid_linear_system.h
* @brief linear system for implicit integration of InvertibleMPMSolid driver
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_INVERTIBLE_MPM_SOLID_LINEAR_SYSTEM_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_INVERTIBLE_MPM_SOLID_LINEAR_SYSTEM_H_

#include "Physika_Numerics/Linear_System_Solvers/linear_system.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;
template <typename Scalar, int Dim> class InvertibleMPMSolid;
template <typename Scalar> class EnrichedMPMUniformGridGeneralizedVector;

/*
 * InvertibleMPMSolidLinearSystem: linear system for implicit integration of InvertibleMPMSolid
 * x and b are represented by EnrichedMPMUniformGridGeneralizedVector
 */

template <typename Scalar, int Dim>
class InvertibleMPMSolidLinearSystem : public LinearSystem<Scalar>
{
public:
    explicit InvertibleMPMSolidLinearSystem(InvertibleMPMSolid<Scalar, Dim> *invertible_driver);
    virtual ~InvertibleMPMSolidLinearSystem();
    virtual void multiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
    virtual void preconditionerMultiply(const GeneralizedVector<Scalar> &x, GeneralizedVector<Scalar> &result) const;
    virtual Scalar innerProduct(const GeneralizedVector<Scalar> &x, const GeneralizedVector<Scalar> &y) const;
	virtual void filter(GeneralizedVector<Scalar> &x) const; //used to apply Dirichlet BCs
    //set the object to construct the linear system for (each object of InvertibleMPMSolid is solved independently)
    //construct one global linear system for all objects if obj_idx is set to -1 (all objects are solved on one grid)
    void setActiveObject(int obj_idx);
protected:
    //potential energy Hessian acted on an arbitrary increment x_diff
    void energyHessianMultiply(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x_diff,
                               EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const;
    //diagonal of energy Hessian acted on an arbitrary increment x_diff, for jacobi precondition
    void energyHessianDiagonalMultiply(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x_diff,
                                       EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const;
    void jacobiPreconditionerMultiply(const EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &x,
                                      EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> > &result) const;
    //disable default copy
    InvertibleMPMSolidLinearSystem();
    InvertibleMPMSolidLinearSystem(const InvertibleMPMSolidLinearSystem<Scalar, Dim> &linear_system);
    InvertibleMPMSolidLinearSystem<Scalar, Dim>& operator= (const InvertibleMPMSolidLinearSystem<Scalar, Dim> &linear_system);
protected:
    InvertibleMPMSolid<Scalar, Dim> *invertible_mpm_solid_driver_;
    int active_obj_idx_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_MPM_MPM_LINEAR_SYSTEMS_INVERTIBLE_MPM_SOLID_LINEAR_SYSTEM_H_