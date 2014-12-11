/*
 * @file CPDI_update_method.h 
 * @Brief the particle domain update procedure introduced in paper:
 *        "A convected particle domain interpolation technique to extend applicability of
 *         the material point method for problems involving massive deformations"
 *        It's the base class of all update methods derived from CPDI
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

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI_UPDATE_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI_UPDATE_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"

namespace Physika{

template <typename Scalar, int Dim> class CPDIMPMSolid;
template <typename Scalar, int Dim> class GridWeightFunction;
template <typename Scalar, int Dim> class SolidParticle;
template <typename Scalar, int Dim> class Grid;

/*
 * constructor is made protected to prohibit creating objects
 * with Dim other than 2 and 3
 */

template <typename Scalar, int Dim>
class CPDIUpdateMethod
{
protected:
    CPDIUpdateMethod();
    ~CPDIUpdateMethod();
};

/*
 * use partial specialization of class template to define CPDI update
 * method for 2D and 3D
 */

template <typename Scalar>
class CPDIUpdateMethod<Scalar,2>
{
public:
    CPDIUpdateMethod();
    virtual ~CPDIUpdateMethod();
    void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
                 std::vector<std::vector<unsigned int> > &particle_grid_pair_num);
    void updateParticleDomain();
    void setCPDIDriver(CPDIMPMSolid<Scalar,2> *cpdi_driver);
protected:
    void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
                                           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
                                           unsigned int &particle_grid_pair_num);
    //helper function, transform between multi-dimension index of grid and flat version index
    unsigned int flatIndex(const Vector<unsigned int,2> &index, const Vector<unsigned int,2> &dimension) const;
    Vector<unsigned int,2> multiDimIndex(unsigned int flat_index, const Vector<unsigned int,2> &dimension) const;
protected:
    CPDIMPMSolid<Scalar,2> *cpdi_driver_;
};

template <typename Scalar>
class CPDIUpdateMethod<Scalar,3>
{
public:
    CPDIUpdateMethod();
    virtual ~CPDIUpdateMethod();
    void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
                 std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
                                                   std::vector<std::vector<unsigned int> > &particle_grid_pair_num);
    void updateParticleDomain();
    void setCPDIDriver(CPDIMPMSolid<Scalar,3> *cpdi_driver);
protected:
    void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
                                           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
                                           unsigned int &particle_grid_pair_num);
    //helper function, transform between multi-dimension index of grid and flat version index
    unsigned int flatIndex(const Vector<unsigned int,3> &index, const Vector<unsigned int,3> &dimension) const;
    Vector<unsigned int,3> multiDimIndex(unsigned int flat_index, const Vector<unsigned int,3> &dimension) const; 
protected:
    CPDIMPMSolid<Scalar,3> *cpdi_driver_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI_UPDATE_METHOD_H_
