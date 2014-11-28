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
#include <map>
#include <set>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Cartesian_Grids/grid.h"
#include "Physika_Dynamics/MPM/mpm_solid_base.h"

namespace Physika{

template<typename Scalar> class DriverPluginBase;
template<typename Scalar,int Dim> class SolidParticle;
template<typename Scalar,int Dim> class MPMSolidContactMethod;

/*
 * MPMSolid: simulate solid with MPM
 * Uniform grid is used as background grid
 * 
 * Single-valued variable is stored on the grid if no specific contact algorithm is employed
 * Otherwise, multi-valued variable maybe attached to a grid node
 */

template <typename Scalar, int Dim>
class MPMSolid: public MPMSolidBase<Scalar,Dim>
{
public:
    MPMSolid();
    MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file, const Grid<Scalar,Dim> &grid);
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
    //get&&set the value at grid nodes corresponding to each object
    Scalar gridMass(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx) const;
    Vector<Scalar,Dim> gridVelocity(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx) const;
    void setGridVelocity(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx, const Vector<Scalar,Dim> &node_velocity);
    //grid nodes used as dirichlet boundary condition, velocity is prescribed
    void addDirichletGridNode(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx);  
    void addDirichletGridNodes(unsigned int object_idx, const std::vector<Vector<unsigned int,Dim> > &node_idx);
    //set contact method
    void setContactMethod(const MPMSolidContactMethod<Scalar,Dim> &contact_method);
    void resetContactMethod();  //reset the contact method to the one inherent in mpm

    //substeps in one time step
    virtual void rasterize();
    virtual void solveOnGrid(Scalar dt);
    virtual void resolveContactOnGrid(Scalar dt);
    virtual void resolveContactOnParticles(Scalar dt);
    virtual void updateParticleInterpolationWeight();
    virtual void updateParticleConstitutiveModelState(Scalar dt);
    virtual void updateParticleVelocity();
    virtual void applyExternalForceOnParticles(Scalar dt);
    virtual void updateParticlePosition(Scalar dt);
    
protected:
    virtual void synchronizeGridData(); //synchronize grid data as grid changes, e.g., size of grid_mass_
    virtual void resetGridData();  //reset grid data to zero, needed before rasterize operation
    virtual Scalar minCellEdgeLength() const;
    virtual void applyGravityOnGrid(Scalar dt);
    virtual void synchronizeWithInfluenceRangeChange(); //synchronize data when the influence range of weight function changes
    bool isValidGridNodeIndex(const Vector<unsigned int,Dim> &node_idx) const;  //helper method, determine if input grid node index is valid
    //manage data attached to particles to stay up-to-date with the particles
    virtual void appendAllParticleRelatedDataOfLastObject();
    virtual void appendLastParticleRelatedDataOfObject(unsigned int object_idx);
    virtual void deleteAllParticleRelatedDataOfObject(unsigned int object_idx);
    virtual void deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx);
    //solve on grid with different integration methods, called in solveOnGrid()
    virtual void solveOnGridForwardEuler(Scalar dt);
    virtual void solveOnGridBackwardEuler(Scalar dt);
    //helper method: conversion between a multi-dimensional grid index and its flat version
    unsigned int flatIndex(const Vector<unsigned int,Dim> &index, const Vector<unsigned int,Dim> &dimension) const;
    Vector<unsigned int,Dim> multiDimIndex(unsigned int flat_index, const Vector<unsigned int,Dim> &dimension) const;
protected:
    Grid<Scalar,Dim> grid_;
    MPMSolidContactMethod<Scalar,Dim> *contact_method_;
    //grid data stored on grid nodes
    //data at each node is a map whose element is the [object_idx, value] pair corresponding to the objects that occupy the node
    ArrayND<std::set<unsigned int>,Dim> is_dirichlet_grid_node_;  //for each grid node, store the object id if it's set as boundary condition for the object
    ArrayND<std::map<unsigned int,Scalar>,Dim> grid_mass_;
    ArrayND<std::map<unsigned int,Vector<Scalar,Dim> >,Dim> grid_velocity_; //current grid velocity
    ArrayND<std::map<unsigned int,Vector<Scalar,Dim> >,Dim> grid_velocity_before_; //grid velocity before any solve update
    std::multimap<unsigned int,unsigned int> active_grid_node_; //the key is the flattened node index, the value is the object id
    //precomputed weights and gradients for grid nodes that is within range of each particle
    //for each particle of each object, store the node-value pair: [object_idx][particle_idx][pair_idx]
    std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > > > particle_grid_weight_and_gradient_;
    std::vector<std::vector<unsigned int> > particle_grid_pair_num_; //the number of pairs in particle_grid_weight_and_gradient_ 
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_MPM_MPM_SOLID_H_
