/*
 * @file invertible_mpm_solid.h 
 * @Brief a hybrid of FEM and CPDI2 for large deformation and invertible elasticity, uniform grid.
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

#ifndef PHYSIKA_DYNAMICS_MPM_INVERTIBLE_MPM_SOLID_H_
#define PHYSIKA_DYNAMICS_MPM_INVERTIBLE_MPM_SOLID_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/MPM/CPDI_mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar, int Dim> class ArrayND;
template <typename Scalar, int Dim> class SquareMatrix;

/*
 * InvertibleMPMSolid: hybrid of FEM and CPDI2 for large deformation and invertible elasticity
 * object number and particle number cannot be changed during run-time
 *
 */

template <typename Scalar, int Dim>
class InvertibleMPMSolid: public CPDIMPMSolid<Scalar,Dim>
{
public:
    InvertibleMPMSolid();
    InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                       const Grid<Scalar,Dim> &grid);
    virtual ~InvertibleMPMSolid();
    
    //restart support
    virtual bool withRestartSupport() const;
    virtual void write(const std::string &file_name);
    virtual void read(const std::string &file_name);

    //virtual methods
    virtual void initSimulationData();  //the topology of the particle domains will be initiated before simulation starts
    virtual void rasterize(); //according to the particle type, some data are rasterized to grid, others to domain corners
    virtual void updateParticleInterpolationWeight();  //interpolation weight between particle and domain corners need to be updated as well
    virtual void updateParticleConstitutiveModelState(Scalar dt);
    virtual void updateParticleVelocity();
    virtual void updateParticlePosition(Scalar dt);
    //explicitly set current particle domain, data in particle_domain_mesh_ are updated as well
    virtual void setCurrentParticleDomain(unsigned int object_idx, unsigned int particle_idx,
                                          const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner);
    void setPrincipalStretchThreshold(Scalar threshold); //set the threshold of principal stretch, value under which will be clamped
protected:
    //solve on grid is reimplemented
    virtual void solveOnGridForwardEuler(Scalar dt);
    virtual void solveOnGridBackwardEuler(Scalar dt);
    virtual void appendAllParticleRelatedDataOfLastObject();
    virtual void appendLastParticleRelatedDataOfObject(unsigned int object_idx);
    virtual void deleteAllParticleRelatedDataOfObject(unsigned int object_idx);
    virtual void deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx);
    virtual void resetParticleDomainData(); //needed before rasterization
    void constructParticleDomainMesh(); //construct particle domain topology from the particle domain positions
    void clearParticleDomainMesh();  //clear memory of particle domain mesh
    bool isEnrichCriteriaSatisfied(unsigned int obj_idx, unsigned int particle_idx) const;  //determine if the particle needs enrichment
    void updateParticleDomainEnrichState();
    void applyGravityOnEnrichedDomainCorner(Scalar dt);
    //the diagonalization technique introduced in "Invertible Finite Elements for Robust Simulation of Large Deformation"
    //trait methods for different dimension
	void diagonalizeDeformationGradient(const SquareMatrix<Scalar,2> &deform_grad, SquareMatrix<Scalar,2> &left_rotation,
		                                SquareMatrix<Scalar,2> &diag_deform_grad, SquareMatrix<Scalar,2> &right_rotation) const;
	void diagonalizeDeformationGradient(const SquareMatrix<Scalar,3> &deform_grad, SquareMatrix<Scalar,3> &left_rotation,
		                                SquareMatrix<Scalar,3> &diag_deform_grad, SquareMatrix<Scalar,3> &right_rotation) const;

protected:
    //for each object, store one volumetric mesh to represent the topology of particle domains
    //each element corresponds to one particle domain
    std::vector<VolumetricMesh<Scalar,Dim>*> particle_domain_mesh_;
    //data attached to each domain corner (vertex of volumetric mesh element), 
    std::vector<std::vector<unsigned char> > is_enriched_domain_corner_;  //use one byte to indicate whether it's enriched or not
    std::vector<std::vector<Scalar> > domain_corner_mass_;
    std::vector<std::vector<Vector<Scalar,Dim> > > domain_corner_velocity_;
    std::vector<std::vector<Vector<Scalar,Dim> > > domain_corner_velocity_before_;
    //interpolation weight between particle and the domain corners, data attached to particle
    std::vector<std::vector<std::vector<Scalar> > > particle_corner_weight_;
    std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > > particle_corner_gradient_;
    //for invertibility support, stretch below this threshold will be clamped to this value
    Scalar principal_stretch_threshold_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_INVERTIBLE_MPM_SOLID_H_
