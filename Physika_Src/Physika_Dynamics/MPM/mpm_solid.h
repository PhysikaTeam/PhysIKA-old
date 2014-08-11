/*
 * @file mpm_solid.h 
 * @Brief MPM driver used to simulate solid, uniform grid.
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

#include <string>
#include <vector>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/MPM/mpm_solid_base.h"

namespace Physika{

template<typename Scalar> class DriverPluginBase;
template<typename Scalar,int Dim> class SolidParticle;

/*
 * MPMSolid: simulate solid with MPM
 * Uniform grid is used as background grid
 */

template <typename Scalar, int Dim>
class MPMSolid: public MPMSolidBase<Scalar,Dim>
{
public:
    MPMSolid();
    MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
             const std::vector<SolidParticle<Scalar,Dim>*> &particles, const Grid<Scalar,Dim> &grid);
    virtual ~MPMSolid();

    //virtual methods
    virtual void initConfiguration(const std::string &file_name);
    virtual void printConfigFileFormat();
    virtual void initSimulationData();
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin);
    virtual bool withRestartSupport() const;
    virtual void write(const std::string &file_name);
    virtual void read(const std::string &file_name);
    
    //setters&&getters
    const Grid<Scalar,Dim>& grid() const;
    void setGrid(const Grid<Scalar,Dim> &grid);
    Scalar gridMass(const Vector<unsigned int,Dim> &node_idx) const;
    Vector<Scalar,Dim> gridVelocity(const Vector<unsigned int,Dim> &node_idx) const;

    //substeps in one time step
    virtual void rasterize();
    virtual void solveOnGrid(Scalar dt);
    virtual void performGridCollision(Scalar dt);
    virtual void performParticleCollision(Scalar dt);
    virtual void updateParticleInterpolationWeight();
    virtual void updateParticleConstitutiveModelState(Scalar dt);
    virtual void updateParticleVelocity();
    virtual void updateParticlePosition(Scalar dt);

protected:
    virtual void synchronizeGridData(); //synchronize grid data as data changes, e.g., size of grid_mass_
    virtual void resetGridData();  //reset grid data to zero, needed before rasterize operation
    virtual Scalar minCellEdgeLength() const;
    virtual void applyGravityOnGrid(Scalar dt);
    bool isValidGridNodeIndex(const Vector<unsigned int,Dim> &node_idx) const;  //helper method, determine if input grid node index is valid
protected:
    Grid<Scalar,Dim> grid_;
    //grid data stored on grid nodes
    std::vector<Vector<unsigned int,Dim> > active_grid_node_; //index of the grid nodes that is active
    ArrayND<Scalar,Dim> grid_mass_;
    ArrayND<Vector<Scalar,Dim>,Dim> grid_velocity_; //current grid velocity
    ArrayND<Vector<Scalar,Dim>,Dim> grid_velocity_before_; //grid velocity before any solve update
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_SOLID_H_
