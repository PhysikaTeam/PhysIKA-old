/*
 * @file CPDI2_update_method.h 
 * @Brief the particle domain update procedure introduced in paper:
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

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI2_UPDATE_METHOD_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI2_UPDATE_METHOD_H_

#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/MPM/mpm_internal.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI_update_method.h"

namespace Physika{

template <typename Scalar, int Dim> class GridWeightFunction;
template <typename ElementType, int Dim> class ArrayND;
template <typename Scalar, int Dim> class VolumetricMesh;

/*
 * Changes compared to conventional CPDI2 in the paper:
 * 1. integration over particle domain is conducted in initial particle domain
 *     instead of in current particle domain
 * 2. the gradient of weight function could be optionally set with respect to reference
 *     configuration, because current particle domain might be degerated and the
 *     gradient is not well defined
 *     
 */


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
    void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,2> &weight_function,
           std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
           std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
           std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
           std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
           bool gradient_to_reference_coordinate = false);
    
    //update the interpolation weight with enrichment
    void updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,2> &weight_function,
           const std::vector<VolumetricMesh<Scalar,2>*> &particle_domain_mesh,
           const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
           std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > > > &particle_grid_weight_and_gradient,
           std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
           std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
           std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
           bool gradient_to_reference_coordinate = false);
    
    //update particle domain with velocity on grid
    void updateParticleDomain(
           const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > > > &corner_grid_weight,
           const std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num, Scalar dt);
    
    //CPDI2 updates particle position according to corner positions
    void updateParticlePosition(Scalar dt, const std::vector<std::vector<unsigned char> > &is_dirichlet_particle);

    //modified CPDI2: compute particle deformation gradient with the displacement of domain corners
    //the deformation gradient of particle is the average of the integrated deformation gradient inside the domain
    //the integration is in initial particle domain, for robustness
    SquareMatrix<Scalar,2> computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx);
   
     //evaluate the deformation gradient of a given point inside the particle domain
    //the given point is expressed as natural coordinate inside the primitive particle domain
    SquareMatrix<Scalar,2> computeDeformationGradientAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, const Vector<Scalar,2> &point_natural_coordinate);

    //compute the element shape function gradient with respect to reference configuration at a given point inside the particle domain
    //the given point is expressed as natural coordinate inside the primitive particle domain
    Vector<Scalar,2> computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                              const Vector<unsigned int,2> &corner_idx, const Vector<Scalar,2> &point_natural_coordinate);

    //the jacobian matrix between the reference particle domain and the primitive one (expressed in natural coordinate)
    SquareMatrix<Scalar,2> computeJacobianBetweenReferenceAndPrimitiveParticleDomain(unsigned int obj_idx, unsigned int particle_idx, const Vector<Scalar,2> &point_natural_coordinate);

    //compute particle interpolation weight/gradient in particle domain
    //the weight/gradient of particle is the average of the integrated weight/gradient inside the domain
    //the integration is in initial particle domain, for robustness
    void computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight);
    void computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Vector<Scalar,2> > &particle_corner_gradient);
protected:
    void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
           unsigned int &particle_grid_pair_num,
           std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > &corner_grid_weight,
           std::vector<unsigned int> &corner_grid_pair_num,
           bool gradient_to_reference_coordinate);
    void updateParticleInterpolationWeightWithEnrichment(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,2> &weight_function,
           const VolumetricMesh<Scalar,2>* particle_domain_mesh,
           const std::vector<unsigned char> &is_enriched_domain_corner,
           std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,2> > &particle_grid_weight_and_gradient,
           unsigned int &particle_grid_pair_num,
           std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,2> > > &corner_grid_weight,
           std::vector<unsigned int> &corner_grid_pair_num,
           bool gradient_to_reference_coordinate);
    //approximate integration of element shape function gradient over particle domain, using 2x2 Gauss integration points
    //the integration domain is the INITIAL particle domain, instead of current particle domain as in the paper
    Vector<Scalar,2> gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(const Vector<unsigned int,2> &corner_idx,
                                                                                            const ArrayND<Vector<Scalar,2>,2> &particle_domain,
                                                                                            const ArrayND<Vector<Scalar,2>,2> &initial_particle_domain);
    Vector<Scalar,2> gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(const Vector<unsigned int,2> &corner_idx,
                                                                                            const ArrayND<Vector<Scalar,2>,2> &initial_particle_domain);
    //the jacobian matrix between particle domain expressed in cartesian coordinate and natural coordinate, evaluated at a point represented in natural coordinate
    //derivative with respect to vector is represented as row vector
    SquareMatrix<Scalar,2> particleDomainJacobian(const Vector<Scalar,2> &eval_point, const ArrayND<Vector<Scalar,2>,2> &particle_domain);
};

template <typename Scalar>
class CPDI2UpdateMethod<Scalar,3>: public CPDIUpdateMethod<Scalar,3>
{
public:
    CPDI2UpdateMethod();
    virtual ~CPDI2UpdateMethod();
    //overwrite methods in CPDIUpdateMethod
    void updateParticleInterpolationWeight(const GridWeightFunction<Scalar,3> &weight_function,
         std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
         std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
         std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
         std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
         bool gradient_to_reference_coordinate = false);

    //update the interpolation weight with enrichment
    void updateParticleInterpolationWeightWithEnrichment(const GridWeightFunction<Scalar,3> &weight_function,
         const std::vector<VolumetricMesh<Scalar,3>*> &particle_domain_mesh,
         const std::vector<std::vector<unsigned char> > &is_enriched_domain_corner,
         std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > > > &particle_grid_weight_and_gradient,
         std::vector<std::vector<unsigned int> > &particle_grid_pair_num,
         std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
         std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num,
         bool gradient_to_reference_coordinate = false);

    //update particle domain with velocity on grid
    void updateParticleDomain(
         const std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > > > &corner_grid_weight,
         const std::vector<std::vector<std::vector<unsigned int> > > &corner_grid_pair_num, Scalar dt);

    //CPDI2 updates particle position according to corner positions
    void updateParticlePosition(Scalar dt, const std::vector<std::vector<unsigned char> > &is_dirichlet_particle);

    //modified CPDI2: compute particle deformation gradient with the displacement of domain corners
    //the deformation gradient of particle is the average of the integrated deformation gradient inside the domain
    //the integration is in initial particle domain, for robustness
    SquareMatrix<Scalar,3> computeParticleDeformationGradientFromDomainShape(unsigned int obj_idx, unsigned int particle_idx);
    
    //evaluate the deformation gradient of a given point inside the particle domain
    //the given point is expressed as natural coordinate inside the primitive particle domain
    SquareMatrix<Scalar,3> computeDeformationGradientAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, const Vector<Scalar,3> &point_natural_coordinate);

    //compute the element shape function gradient with respect to reference configuration at a given point inside the particle domain
    //the given point is expressed as natural coordinate inside the primitive particle domain
    Vector<Scalar,3> computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(unsigned int obj_idx, unsigned int particle_idx,
                                                                                              const Vector<unsigned int,3> &corner_idx, const Vector<Scalar,3> &point_natural_coordinate);

    //the jacobian matrix between the reference particle domain and the primitive one (expressed in natural coordinate)
    SquareMatrix<Scalar,3> computeJacobianBetweenReferenceAndPrimitiveParticleDomain(unsigned int obj_idx, unsigned int particle_idx, const Vector<Scalar,3> &point_natural_coordinate);

    //compute particle interpolation weight/gradient in particle domain
    //the weight/gradient of particle is the average of the integrated weight/gradient inside the domain
    //the integration is in initial particle domain, for robustness
    void computeParticleInterpolationWeightInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Scalar> &particle_corner_weight);
    void computeParticleInterpolationGradientInParticleDomain(unsigned int obj_idx, unsigned int particle_idx, std::vector<Vector<Scalar,3> > &particle_corner_gradient);
    
protected:
    void updateParticleInterpolationWeight(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
         std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
         unsigned int &particle_grid_pair_num,
         std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > &corner_grid_weight,
         std::vector<unsigned int> &corner_grid_pair_num,
         bool gradient_to_reference_coordinate);
    void updateParticleInterpolationWeightWithEnrichment(unsigned int object_idx, unsigned int particle_idx, const GridWeightFunction<Scalar,3> &weight_function,
         const VolumetricMesh<Scalar,3>* particle_domain_mesh,
         const std::vector<unsigned char> &is_enriched_domain_corner,
         std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,3> > &particle_grid_weight_and_gradient,
         unsigned int &particle_grid_pair_num,
         std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,3> > > &corner_grid_weight,
         std::vector<unsigned int> &corner_grid_pair_num,
         bool gradient_to_reference_coordinate);
    //approximate integration of element shape function (gradient) over particle domain, using 2x2x2 Gauss integration points
    //the integration domain is the INITIAL particle domain, instead of current particle domain as in the paper
    Scalar gaussIntegrateShapeFunctionValueInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain);
    Vector<Scalar,3> gaussIntegrateShapeFunctionGradientToCurrentCoordinateInParticleDomain(const Vector<unsigned int,3> &corner_idx, const ArrayND<Vector<Scalar,3>,3> &particle_domain,
                                                                                            const ArrayND<Vector<Scalar,3>,3> &initial_particle_domain);
    Vector<Scalar,3> gaussIntegrateShapeFunctionGradientToReferenceCoordinateInParticleDomain(const Vector<unsigned int,3> &corner_idx,
                                                                                            const ArrayND<Vector<Scalar,3>,3> &initial_particle_domain);
    //the jacobian matrix between particle domain expressed in cartesian coordinate and natural coordinate, evaluated at a point represented in natural coordinate
    //derivative with respect to vector is represented as row vector
    SquareMatrix<Scalar,3> particleDomainJacobian(const Vector<Scalar,3> &eval_point, const ArrayND<Vector<Scalar,3>,3> &particle_domain);
    //compute the volume of given particle domain
    Scalar particleDomainVolume(const ArrayND<Vector<Scalar,3>,3> &particle_domain);
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_UPDATE_METHODS_CPDI2_UPDATE_METHOD_H_
