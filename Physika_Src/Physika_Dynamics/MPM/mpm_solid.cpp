/*
 * @file mpm_solid.cpp
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

#include <cstdlib>
#include <limits>
#include <iostream>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Driver/driver_plugin_base.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_step_method.h"
#include "Physika_Dynamics/MPM/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid()
    :MPMSolidBase<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :MPMSolidBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                               const std::vector<SolidParticle<Scalar,Dim>*> &particles, const Grid<Scalar,Dim> &grid)
    :MPMSolidBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,particles),grid_(grid)
{
    synchronizeGridData();
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::~MPMSolid()
{
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::initConfiguration(const std::string &file_name)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::printConfigFileFormat()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::initSimulationData()
{
    resetGridData();
    updateParticleInterpolationWeight();//initialize the interpolation weight before simulation
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{
    if(plugin==NULL)
    {
        std::cerr<<"Warning: NULL plugin provided, operation ignored!\n";
        return;
    }

    if(dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(plugin)==NULL)
    {
        std::cerr<<"Warning: Wrong type of plugin provided, operation ignored!\n";
        return;
    }
    plugin->setDriver(this);
    this->plugins_.push_back(plugin);
}

template <typename Scalar, int Dim>
bool MPMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::write(const std::string &file_name)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::read(const std::string &file_name)
{
//TO DO
}

template <typename Scalar, int Dim>
const Grid<Scalar,Dim>& MPMSolid<Scalar,Dim>::grid() const
{
    return grid_;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::setGrid(const Grid<Scalar,Dim> &grid)
{
    grid_ = grid;
    synchronizeGridData();
}

template <typename Scalar, int Dim>
Scalar MPMSolid<Scalar,Dim>::gridMass(const Vector<unsigned int,Dim> &node_idx) const
{
    bool valid_idx = isValidGridNodeIndex(node_idx);
    if(!valid_idx)
    {
        std::cerr<<"Error: invalid node index, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    return grid_mass_(node_idx);
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> MPMSolid<Scalar,Dim>::gridVelocity(const Vector<unsigned int,Dim> &node_idx) const
{
    bool valid_idx = isValidGridNodeIndex(node_idx);
    if(!valid_idx)
    {
        std::cerr<<"Error: invalid node index, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    return grid_velocity_(node_idx);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::rasterize()
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onRasterize();
    }

    //rasterize mass and momentum to grid
    resetGridData();
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        unsigned int j = 0;
        for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++j,++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Scalar weight = this->particle_grid_weight_[i][j]; 
            grid_mass_(node_idx) += weight*particle->mass();
            grid_velocity_(node_idx) += weight*(particle->mass()*particle->velocity());
        }
    }
    //determine active grid nodes according to the grid mass
    for(typename ArrayND<Scalar,Dim>::Iterator iter = grid_mass_.begin(); iter != grid_mass_.end(); ++iter)
    {
        Vector<unsigned int,Dim> ele_idx = iter.elementIndex();
        if(grid_mass_(ele_idx)>std::numeric_limits<Scalar>::epsilon())
            active_grid_node_.push_back(ele_idx);
    }
    //compute grid's velocity, divide momentum by mass
    for(unsigned int i = 0; i < active_grid_node_.size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = active_grid_node_[i];
        grid_velocity_(node_idx) /= grid_mass_(node_idx);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::solveOnGrid(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onSolveOnGrid(dt);
    }

    //only explicit integration is implemented yet
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        unsigned j = 0;
        for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++j,++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> weight_gradient = this->particle_grid_weight_gradient_[i][j];
            SquareMatrix<Scalar,Dim> cauchy_stress = particle->cauchyStress();
            if(grid_mass_(node_idx)>std::numeric_limits<Scalar>::epsilon())
                grid_velocity_(node_idx) += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/grid_mass_(node_idx);
        }
    }
    //apply gravity
    applyGravityOnGrid(dt);
    //TO DO (implicit)
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::performGridCollision(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onPerformGridCollision(dt);
    }
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::performParticleCollision(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onPerformParticleCollision(dt);
    }
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticleInterpolationWeight()
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleInterpolationWeight();
    }

    //first resize the vectors
    (this->particle_grid_weight_).resize(this->particles_.size());
    (this->particle_grid_weight_gradient_).resize(this->particles_.size());
    //precompute the interpolation weights and gradients
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> grid_dx = (this->grid_).dX();
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        (this->particle_grid_weight_)[i].clear();
        (this->particle_grid_weight_gradient_)[i].clear();
        for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
            for(unsigned int dim = 0; dim < Dim; ++dim)
                particle_to_node[dim] /= grid_dx[dim];
            Vector<Scalar,Dim> weight_gradient = this->weight_function_->gradient(particle_to_node);
            Scalar weight = this->weight_function_->weight(particle_to_node);
            (this->particle_grid_weight_)[i].push_back(weight);
            (this->particle_grid_weight_gradient_)[i].push_back(weight_gradient);
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleConstitutiveModelState(dt);
    }

    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        SquareMatrix<Scalar,Dim> particle_vel_grad(0);
        unsigned int j = 0;
        for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++j,++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> weight_gradient = this->particle_grid_weight_gradient_[i][j];
            particle_vel_grad += grid_velocity_(node_idx).outerProduct(weight_gradient);
        }
        SquareMatrix<Scalar,Dim> particle_deform_grad = particle->deformationGradient();
        particle_deform_grad += dt*particle_vel_grad*particle_deform_grad;
        particle->setDeformationGradient(particle_deform_grad);  //update deformation gradient
        Scalar particle_vol = (particle_deform_grad.determinant())*(particle->volume());
        particle->setVolume(particle_vol);  //update particle volume
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticleVelocity()
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleVelocity();
    }

    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    //direct interpolate grid velocity to particle
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        Vector<Scalar,Dim> new_vel(0);
        unsigned int j = 0;
        for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++j,++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Scalar weight = this->particle_grid_weight_[i][j]; 
            new_vel += weight*grid_velocity_(node_idx);
        }
        particle->setVelocity(new_vel);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticlePosition(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticlePosition(dt);
    }

    //update particle's position with the new velocity
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> new_pos = particle->position() + particle->velocity()*dt;
        particle->setPosition(new_pos);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::synchronizeGridData()
{
    Vector<unsigned int,Dim> node_num = grid_.nodeNum();
    for(unsigned int i = 0; i < Dim; ++i)
    {
        grid_mass_.resize(node_num[i],i);
        grid_velocity_.resize(node_num[i],i);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::resetGridData()
{
    active_grid_node_.clear();
    for(typename Grid<Scalar,Dim>::NodeIterator iter = grid_.nodeBegin(); iter != grid_.nodeEnd(); ++iter)
    {
        Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
        std::vector<unsigned int> node_idx_vec;
        for(unsigned int i = 0; i < Dim; ++i)
            node_idx_vec.push_back(node_idx[i]);
        grid_mass_(node_idx_vec) = 0;
        grid_velocity_(node_idx_vec) = Vector<Scalar,Dim>(0);
    }
}

template <typename Scalar, int Dim>
Scalar MPMSolid<Scalar,Dim>::minCellEdgeLength() const
{
    return grid_.minEdgeLength();
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::applyGravityOnGrid(Scalar dt)
{
    //apply gravity on active grid node
    for(unsigned int i = 0; i < active_grid_node_.size(); ++i)
    {
        Vector<unsigned int,Dim> node_idx = active_grid_node_[i];
        Vector<Scalar,Dim> gravity_vec(0);
        gravity_vec[1] = (-1)*(this->gravity_);
        grid_velocity_(node_idx) += gravity_vec*dt;
    }
}

template <typename Scalar, int Dim>
bool MPMSolid<Scalar,Dim>::isValidGridNodeIndex(const Vector<unsigned int,Dim> &node_idx) const
{
    Vector<unsigned int,Dim> node_num = grid_.nodeNum();
    for(unsigned int i = 0; i < Dim; ++i)
        if(node_idx[i] >= node_num[i])
            return false;
    return true;
}

//explicit instantiations
template class MPMSolid<float,2>;
template class MPMSolid<float,3>;
template class MPMSolid<double,2>;
template class MPMSolid<double,3>;

}  //end of namespace Physika
