/*
 * @file robust_CPDI2_update_method.h 
 * @brief enhanced version of the particle domain update procedure introduced in paper:
 *        "Second-order convected particle domain interpolation with enrichment for weak
 *         discontinuities at material interfaces"
 *    We made some key modifications(enhancements) to the conventional CPDI2 to improve
 *    its robustness with degenerated particle domain during simulation
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

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_ROBUST_CPDI2_UPDATE_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_ROBUST_CPDI2_UPDATE_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"

namespace Physika{

template <typename Scalar, int Dim> class GridWeightFunction;
template <typename ElementType, int Dim> class ArrayND;
template <typename Scalar, int Dim> class VolumetricMesh;

/*
 * Changes compared to conventional CPDI2 in the paper:
 * 1. integration over particle domain is conducted in initial particle domain
 *     instead of in current particle domain
 * 2. the gradient of weight function is with respect to reference
 *     configuration, because current particle domain might be degerated and the
 *     gradient is not well defined
 * 
 *  unless otherwise noted, the volume integration and spatial gradient are with
 *  respect to reference configuration    
 */

/*
 * constructor is made protected to prohibit creating objects
 * with Dim other than 2 and 3
 */

template <typename Scalar, int Dim>
class RobustCPDI2UpdateMethod: public CPDI2UpdateMethod<Scalar,Dim>
{
protected:
    RobustCPDI2UpdateMethod();
    ~RobustCPDI2UpdateMethod();
};

/*
 * use partial specialization of class template to define CPDI2 update
 * method for 2D and 3D
 */

template <typename Scalar>
class RobustCPDI2UpdateMethod<Scalar,2>: public CPDI2UpdateMethod<Scalar,2>
{
public:
    RobustCPDI2UpdateMethod();
    virtual ~RobustCPDI2UpdateMethod();

    virtual void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
           std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
           std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
           std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
           std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);
    //faster version of updateParticleInterpolationWeight, remove redundent computation of corner-grid weight by providing topology strcuture of
    //particle domains
    virtual void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
           const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
           std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
           std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
           std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
           std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);

    //update the interpolation weight with enrichment
    virtual void updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,2> &weight_function,
           const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
           const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
           std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
           std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
           std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
           std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);

    //compute particle deformation gradient with the displacement of domain corners
    //the deformation gradient of particle is the average of the integrated deformation gradient inside the domain
    //the integration is in initial particle domain, for robustness
    virtual SquareMatrix<Scalar,2> computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx);

    //compute particle interpolation weight/gradient in particle domain
    //the weight/gradient of particle is the average of the integrated weight/gradient inside the domain
    //the integration is in initial particle domain, for robustness
    virtual void computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight);
    virtual void computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Vector<Scalar,2> > &particle_corner_gradient);
protected:
    void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
           unsigned int &particle_grid_pair_num,
           std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > &corner_grid_weight,
           std::vector<unsigned int> &corner_grid_pair_num);
    //approximate integration of element shape function gradient over particle domain, using 2x2 Gauss integration points
    //the integration domain is the INITIAL particle domain, instead of current particle domain as in the paper
    Vector<Scalar,2> gaussIntegrateShapeFunctionGradientInParticleDomain(const Vector<unsigned int,2> &corner_idx,
                                                                         const ArrayND<Vector<Scalar,2>,2> &initial_particle_domain);
};

template <typename Scalar>
class RobustCPDI2UpdateMethod<Scalar,3>: public CPDI2UpdateMethod<Scalar,3>
{
public:
    RobustCPDI2UpdateMethod();
    virtual ~RobustCPDI2UpdateMethod();

    virtual void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
         std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
         std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
         std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
         std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);
    //faster version of updateParticleInterpolationWeight, remove redundent computation of corner-grid weight by providing topology strcuture of
    //particle domains
    virtual void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
           const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
           std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
           std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
           std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
           std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);

    //update the interpolation weight with enrichment
    virtual void updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,3> &weight_function,
         const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
         const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
         std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
         std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
         std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
         std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num);

    //compute particle deformation gradient with the displacement of domain corners
    //the deformation gradient of particle is the average of the integrated deformation gradient inside the domain
    //the integration is in initial particle domain, for robustness
    virtual SquareMatrix<Scalar,3> computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx);

    //compute particle interpolation weight/gradient in particle domain
    //the weight/gradient of particle is the average of the integrated weight/gradient inside the domain
    //the integration is in initial particle domain, for robustness
    virtual void computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight);
    virtual void computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Vector<Scalar,3> > &particle_corner_gradient);
    
protected:
    void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
         std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
         unsigned int &particle_grid_pair_num,
         std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > &corner_grid_weight,
         std::vector<unsigned int> &corner_grid_pair_num);
    //approximate integration of element shape function (gradient) over particle domain, using 2x2x2 Gauss integration points
    //the integration domain is the INITIAL particle domain, instead of current particle domain as in the paper
    Scalar gaussIntegrateShapeFunctionValueInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain);
    Vector<Scalar,3> gaussIntegrateShapeFunctionGradientInParticleDomain(const Vector<unsigned int,3> &corner_idx,
                                                                                            const ArrayND<Vector<Scalar,3>,3> &initial_particle_domain);
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_ROBUST_CPDI2_UPDATE_METHOD_H_
