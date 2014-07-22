/*
 * @file mpm_solid.h 
 * @Brief MPM driver used to simulate solid.
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

#ifndef PHYSIKA_DYNAMICS_MPM_MPM_SOLID_H_
#define PHYSIKA_DYNAMICS_MPM_MPM_SOLID_H_

#include <vector>
#include <string>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/Driver/driver_base.h"

namespace Physika{

template<typename Scalar> class DriverPluginBase;
template<typename Scalar,int Dim> class SolidParticle;

/*
 * MPMSolid: simulate solid with MPM
 * Uniform grid is used as background grid
 */

template <typename Scalar, int Dim>
class MPMSolid: public DriverBase<Scalar>
{
public:
    MPMSolid();
    MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
             const std::vector<SolidParticle<Scalar,Dim>*> &particles, const Grid<Scalar,Dim> &grid);
    virtual ~MPMSolid();

    //virtual methods
    virtual void initConfiguration(const std::string &file_name);
    virtual void advanceStep(Scalar dt);
    virtual Scalar computeTimeStep();
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin);
    virtual bool withRestartSupport() const;
    virtual void write(const std::string &file_name);
    virtual void read(const std::string &file_name);

    //get && set
    unsigned int particleNum() const;
    void addParticle(const SolidParticle<Scalar,Dim> &particle);
    void removeParticle(unsigned int particle_idx);
    void setParticles(const std::vector<SolidParticle<Scalar,Dim>*> &particles); //set all simulation particles, data are copied
    const SolidParticle<Scalar,Dim>& particle(unsigned int particle_idx) const;
    SolidParticle<Scalar,Dim>& particle(unsigned int particle_idx);
    const Grid<Scalar,Dim>& grid() const;
    void setGrid(const Grid<Scalar,Dim> &grid);

protected:
    virtual void initialize();
    void synchronizeGridData(); //synchronize grid data as data changes, e.g., size of grid_mass_
    //substeps in one time step
    void rasterize();
    void updateGridVelocity();
    void performGridCollision();
    void performParticleCollision();
    void updateParticleInterpolationWeight();
    void updateParticleConstitutiveModelState();
    void updateParticlePositionAndVelocity();
protected:
    std::vector<SolidParticle<Scalar,Dim>*> particles_;
    Grid<Scalar,Dim> grid_;
    //grid data stored on grid nodes
    ArrayND<Scalar,Dim> grid_mass_;
    ArrayND<Vector<Scalar,Dim>,Dim> grid_velocity_;
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_SOLID_H_
