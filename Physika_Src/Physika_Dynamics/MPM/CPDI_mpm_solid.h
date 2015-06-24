/*
 * @file CPDI_mpm_solid.h 
 * @Brief CPDI(CPDI2) MPM driver used to simulate solid, uniform grid.
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

#ifndef PHYSIKA_DYNAMICS_MPM_CPDI_MPM_SOLID_H_
#define PHYSIKA_DYNAMICS_MPM_CPDI_MPM_SOLID_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI_update_method.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"

namespace Physika{

template <typename Scalar,int Dim> class SolidParticle;

template <typename Scalar, int Dim>
class CPDIMPMSolid: public MPMSolid<Scalar,Dim>
{
public:
    CPDIMPMSolid();
    CPDIMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    CPDIMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, const Grid<Scalar,Dim> &grid);
    virtual ~CPDIMPMSolid();
    
    //restart support
    virtual bool withRestartSupport() const;
    virtual void write(const std::string &file_name);
    virtual void read(const std::string &file_name);

    //re-implemented methods compared to standard MPM
    virtual void updateParticleInterpolationWeight();  //compute the interpolation weight between particles and grid nodes
    virtual void updateParticlePosition(Scalar dt);

    //return current corners of given particle, empty array is returned if index is invalid
    void currentParticleDomain(unsigned int object_idx, unsigned int particle_idx, ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner) const;
    //explicitly set current particle domain
    virtual void setCurrentParticleDomain(unsigned int object_idx, unsigned int particle_idx, const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner);
    //return initial corners of given particle, empty array is returned if index is invalid
    void initialParticleDomain(unsigned int object_idx, unsigned int particle_idx, ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner) const;
    //explicitly initialize particle domain
    void initializeParticleDomain(unsigned int object_idx, unsigned int particle_idx, const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner);
    //get specific domain corner
    Vector<Scalar,Dim> currentParticleDomainCorner(unsigned int object_idx, unsigned int particle_idx, const Vector<unsigned int,Dim> &corner_idx) const;
    Vector<Scalar,Dim> initialParticleDomainCorner(unsigned int object_idx, unsigned int particle_idx, const Vector<unsigned int,Dim> &corner_idx) const;

    //set the particle domain update method using method update type as template
    template <typename CPDIUpdateMethodType>
    void setCPDIUpdateMethod();

    //save particle domain as Physika::VolumetricMesh
    virtual bool saveParticleDomain(unsigned int object_idx, const std::string &file_name) const;

protected:
    virtual void synchronizeWithInfluenceRangeChange();
    //manage data related to all particles, e.g., precomputed node weight and gradient
    //In CPDI, the grid nodes that influence particles are the ones that are 
    //within influence range of the particle domain corners
    virtual void appendAllParticleRelatedDataOfLastObject();
    virtual void appendLastParticleRelatedDataOfObject(unsigned int object_idx);
    virtual void deleteAllParticleRelatedDataOfObject(unsigned int object_idx);
    virtual void deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx);
    //trait method to init particle domain
    void initParticleDomain(const SolidParticle<Scalar,2> &particle, std::vector<Vector<Scalar,2> > &domain_corner);
    void initParticleDomain(const SolidParticle<Scalar,3> &particle, std::vector<Vector<Scalar,3> > &domain_corner);
protected:
    std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > > particle_domain_corners_;  //current particle domain corners
    std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > > initial_particle_domain_corners_; //initial particle domain corners
    //for each domain corner of the particles, store precomputed weights of grid nodes that is within influence range
    //needed in CPDI2
    std::vector<std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightPair<Scalar,Dim> > > > > corner_grid_weight_;
    std::vector<std::vector<std::vector<unsigned int> > > corner_grid_pair_num_; //the number of pairs for each corner of each particle
    CPDIUpdateMethod<Scalar,Dim> *cpdi_update_method_; //the cpdi method used to update particle domain
};

template <typename Scalar, int Dim>
template <typename CPDIUpdateMethodType>
void CPDIMPMSolid<Scalar,Dim>::setCPDIUpdateMethod()
{
    if(cpdi_update_method_)
        delete cpdi_update_method_;
    cpdi_update_method_ = new CPDIUpdateMethodType();
    cpdi_update_method_->setCPDIDriver(this);
}

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_CPDI_MPM_SOLID_H_
