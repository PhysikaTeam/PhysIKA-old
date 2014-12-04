/*
 * @file CPDI2_update_method.h 
 * @Brief the particle domain update procedure introduced in paper:
 *        "Second-order convected particle domain interpolation with enrichment for weak
 *         discontinuities at material interfaces"
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI2_UPDATE_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI2_UPDATE_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI_update_method.h"

namespace Physika{

template <typename Scalar, int Dim> class GridWeightFunction;
template <typename ElementType, int Dim> class ArrayND;

/*
 * constructor is made protected to prohibit creating objects
 * with Dim other than 2 and 3
 */

template <typename Scalar, int Dim>
class CPDI2UpdateMethod: public CPDIUpdateMethod<Scalar,Dim>
{
protected:
    CPDI2UpdateMethod();
    ~CPDI2UpdateMethod();
};

/*
 * use partial specialization of class template to define CPDI2 update
 * method for 2D and 3D
 */

template <typename Scalar>
class CPDI2UpdateMethod<Scalar,2>: public CPDIUpdateMethod<Scalar,2>
{
public:
    CPDI2UpdateMethod();
    virtual ~CPDI2UpdateMethod();
    //overwrite methods in CPDIUpdateMethod
    virtual void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > > &corner_grid_weight_and_gradient,
                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);

    virtual void updateParticleDomain(
        const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > > &corner_grid_weight_and_gradient,
        const std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num, Scalar dt);
    
    //CPDI2 updates particle position according to corner positions
    virtual void updateParticlePosition(Scalar dt, const std::vector<std::vector<unsigned char> > &is_dirichlet_particle);

    //modified CPDI2: compute particle deformation gradient with the displacement of domain corners
    void updateParticleDeformationGradient(Scalar dt);
protected:
    virtual void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                                   std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                                   unsigned int &particle_grid_pair_num,
                                                   std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > &corner_grid_weight_and_gradient,
                                                   std::vector<unsigned int> &corner_grid_pair_num);
};

template <typename Scalar>
class CPDI2UpdateMethod<Scalar,3>: public CPDIUpdateMethod<Scalar,3>
{
public:
    CPDI2UpdateMethod();
    virtual ~CPDI2UpdateMethod();
    //overwrite methods in CPDIUpdateMethod
    virtual void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
                 std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > > &corner_grid_weight_and_gradient,
                 std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);

    virtual void updateParticleDomain(
        const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > > &corner_grid_weight_and_gradient,
        const std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num, Scalar dt);

    //CPDI2 updates particle position according to corner positions
    virtual void updateParticlePosition(Scalar dt, const std::vector<std::vector<unsigned char> > &is_dirichlet_particle);

    //modified CPDI2: compute particle deformation gradient with the displacement of domain corners
    void updateParticleDeformationGradient(Scalar dt);
protected:
    virtual void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                                   std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                                   unsigned int &particle_grid_pair_num,
                                                   std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > &corner_grid_weight_and_gradient,
                                                   std::vector<unsigned int> &corner_grid_pair_num);
    //approximate integration of element shape function (gradient) over particle domain, using 2x2x2 Gauss integration points
    Scalar gaussianIntegrateShapeFunctionValueInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain);
    Vector<Scalar,3> gaussianIntegrateShapeFunctionGradientInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain);
    //the jacobian matrix between particle domain expressed in cartesian coordinate and natural coordinate, evaluated at a point represented in natural coordinate
    //derivative with respect to vector is represented as row vector
    SquareMatrix<Scalar,3> particleDomainJacobian(const Vector<Scalar,3> &eval_point, const ArrayND<Vector<Scalar,3>,3> &particle_domain);
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI2_UPDATE_METHOD_H_
