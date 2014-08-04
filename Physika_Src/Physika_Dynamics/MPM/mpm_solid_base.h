/*
 * @file mpm_solid_base.h 
 * @Brief base class of all MPM drivers for solid.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_SOLID_BASE_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_SOLID_BASE_H_

#include <string>
#include <vector>
#include "Physika_Dynamics/MPM/mpm_base.h"

namespace Physika{

template<typename Scalar> class DriverPluginBase;
template<typename Scalar,int Dim> class SolidParticle;

template <typename Scalar, int Dim>
class MPMSolidBase: public MPMBase<Scalar,Dim>
{
public:
    MPMSolidBase();
    MPMSolidBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    MPMSolidBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
             const std::vector<SolidParticle<Scalar,Dim>*> &particles);
    virtual ~MPMSolidBase();

    //virtual methods
    virtual void initConfiguration(const std::string &file_name)=0;
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin)=0;
    virtual bool withRestartSupport() const=0;
    virtual void write(const std::string &file_name)=0;
    virtual void read(const std::string &file_name)=0;

    //get && set
    unsigned int particleNum() const;
    void addParticle(const SolidParticle<Scalar,Dim> &particle);
    void removeParticle(unsigned int particle_idx);
    void setParticles(const std::vector<SolidParticle<Scalar,Dim>*> &particles); //set all simulation particles, data are copied
    const SolidParticle<Scalar,Dim>& particle(unsigned int particle_idx) const;
    SolidParticle<Scalar,Dim>& particle(unsigned int particle_idx);

    //substeps in one time step
    virtual void rasterize()=0;  //rasterize data to grid
    virtual void solveOnGrid()=0; //solve the dynamics system on grid
    virtual void performGridCollision()=0;
    virtual void performParticleCollision()=0;
    virtual void updateParticleInterpolationWeight()=0;
    virtual void updateParticleConstitutiveModelState()=0;
    virtual void updateParticleVelocity()=0;
    virtual void updateParticlePosition()=0;
protected:
    virtual void initialize()=0;
    virtual Scalar minCellEdgeLength() const=0; //minimum edge length of the background grid, for dt computation
    virtual Scalar maxParticleVelocityNorm() const;
protected:
    std::vector<SolidParticle<Scalar,Dim>*> particles_;
};

}//namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_SOLID_BASE_H_
