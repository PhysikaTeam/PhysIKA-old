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
#include <algorithm>
#include "Physika_Core/Utilities/physika_assert.h"
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
void MPMSolid<Scalar,Dim>::setGridVelocity(const Vector<unsigned int, Dim> &node_idx, const Vector<Scalar,Dim> &node_velocity)
{
    bool valid_idx = isValidGridNodeIndex(node_idx);
    if(!valid_idx)
    {
        std::cerr<<"Warning: invalid node index, operation ignored!\n";
        return;
    }
    grid_velocity_(node_idx) = node_velocity;
    grid_velocity_before_(node_idx) = node_velocity;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addDirichletGridNode(const Vector<unsigned int,Dim> &node_idx)
{
    bool valid_idx = isValidGridNodeIndex(node_idx);
    if(!valid_idx)
    {
        std::cerr<<"Warning: invalid node index, operation ignored!\n";
        return;
    }
    is_dirichlet_grid_node_(node_idx) = 1;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addDirichletGridNodes(const std::vector<Vector<unsigned int,Dim> > &node_idx)
{
    for(unsigned int i = 0; i < node_idx.size(); ++i)
        addDirichletGridNode(node_idx[i]);
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
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        for(unsigned int j = 0; j < this->particle_grid_pair_num_[i]; ++j)
        {
            const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[i][j];
            Scalar weight = pair.weight_value_; 
            grid_mass_(pair.node_idx_) += weight*particle->mass();
            if(is_dirichlet_grid_node_(pair.node_idx_)) //skip the velocity update of boundary nodes
                continue;
            grid_velocity_(pair.node_idx_) += weight*(particle->mass()*particle->velocity());
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
        if(is_dirichlet_grid_node_(node_idx))
            continue; //skip grid nodes that are boundary condition
        grid_velocity_(node_idx) /= grid_mass_(node_idx);
        grid_velocity_before_(node_idx) = grid_velocity_(node_idx); //buffer the grid velocity before any update
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

    switch (this->integration_method_)
    {
    case MPMSolidBase<Scalar,Dim>::FORWARD_EULER:
        solveOnGridForwardEuler(dt);
        break;
    case MPMSolidBase<Scalar,Dim>::BACKWARD_EULER:
        solveOnGridBackwardEuler(dt);
        break;
    default:
        break;
    }
    //apply gravity
    applyGravityOnGrid(dt);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::resolveContactOnGrid(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onResolveContactOnGrid(dt);
    }
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::resolveContactOnParticles(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onResolveContactOnParticles(dt);
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
    PHYSIKA_ASSERT(this->particle_grid_weight_and_gradient_.size() == this->particles_.size());
    //precompute the interpolation weights and gradients
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> grid_dx = (this->grid_).dX();
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        this->particle_grid_pair_num_[i] = 0;
        for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
            for(unsigned int dim = 0; dim < Dim; ++dim)
                particle_to_node[dim] /= grid_dx[dim];
            Vector<Scalar,Dim> weight_gradient = this->weight_function_->gradient(particle_to_node);
            Scalar weight = this->weight_function_->weight(particle_to_node);
            unsigned int j = this->particle_grid_pair_num_[i]++;
            (this->particle_grid_weight_and_gradient_)[i][j].node_idx_ = node_idx;
            (this->particle_grid_weight_and_gradient_)[i][j].weight_value_ = weight;
            (this->particle_grid_weight_and_gradient_)[i][j].gradient_value_ = weight_gradient;
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

    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        SquareMatrix<Scalar,Dim> particle_vel_grad(0);
        for(unsigned int j = 0; j < this->particle_grid_pair_num_[i]; ++j)
        {
            Vector<unsigned int,Dim> node_idx = (this->particle_grid_weight_and_gradient_[i][j].node_idx_);
            Vector<Scalar,Dim> weight_gradient = (this->particle_grid_weight_and_gradient_[i][j].gradient_value_);
            particle_vel_grad += grid_velocity_(node_idx).outerProduct(weight_gradient);
        }
        SquareMatrix<Scalar,Dim> particle_deform_grad = particle->deformationGradient();
        particle_deform_grad += dt*particle_vel_grad*particle_deform_grad;
        particle->setDeformationGradient(particle_deform_grad);  //update deformation gradient
        Scalar particle_vol = (particle_deform_grad.determinant())*(this->particle_initial_volume_[i]);
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

    //interpolate delta of grid velocity to particle
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        if(this->is_dirichlet_particle_[i])
            continue;//skip boundary particles
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> new_vel = particle->velocity();
        for(unsigned int j = 0; j < this->particle_grid_pair_num_[i]; ++j)
        {
            Vector<unsigned int,Dim> node_idx = this->particle_grid_weight_and_gradient_[i][j].node_idx_;
            Scalar weight = this->particle_grid_weight_and_gradient_[i][j].weight_value_;
            new_vel += weight*(grid_velocity_(node_idx)-grid_velocity_before_(node_idx));
        }
        particle->setVelocity(new_vel);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::applyExternalForceOnParticles(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onApplyExternalForceOnParticles(dt);
    }

    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        if(this->is_dirichlet_particle_[i])
            continue;//skip boundary particles
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> new_vel = particle->velocity();
        new_vel += this->particle_external_force_[i]/particle->mass()*dt;
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

    //update particle's position with the new grid velocity
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> new_pos = particle->position();
        if(this->is_dirichlet_particle_[i]) //for dirichlet particles, update position with prescribed velocity
            new_pos += this->particles_[i]->velocity()*dt;
        else
        {
            for(unsigned int j = 0; j < this->particle_grid_pair_num_[i]; ++j)
            {
                Vector<unsigned int,Dim> node_idx = this->particle_grid_weight_and_gradient_[i][j].node_idx_;
                Scalar weight = this->particle_grid_weight_and_gradient_[i][j].weight_value_;
                new_pos += weight*grid_velocity_(node_idx)*dt;
            }
        }
        particle->setPosition(new_pos);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::synchronizeGridData()
{
    Vector<unsigned int,Dim> node_num = grid_.nodeNum();
    for(unsigned int i = 0; i < Dim; ++i)
    {
        is_dirichlet_grid_node_.resize(node_num[i],i);
        grid_mass_.resize(node_num[i],i);
        grid_velocity_.resize(node_num[i],i);
        grid_velocity_before_.resize(node_num[i],i);
    }
    //initialize boundary condition grid node indicator
    for(typename ArrayND<unsigned char,Dim>::Iterator iter = is_dirichlet_grid_node_.begin(); iter != is_dirichlet_grid_node_.end(); ++iter)
        *iter = 0;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::resetGridData()
{
    active_grid_node_.clear();
    for(typename Grid<Scalar,Dim>::NodeIterator iter = grid_.nodeBegin(); iter != grid_.nodeEnd(); ++iter)
    {
        Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
        grid_mass_(node_idx) = 0;
        if(is_dirichlet_grid_node_(node_idx))
            continue; //skip grid nodes that are boundary condition
        grid_velocity_(node_idx) = Vector<Scalar,Dim>(0);
        grid_velocity_before_(node_idx) = Vector<Scalar,Dim>(0);
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
        if(is_dirichlet_grid_node_(node_idx))
            continue; //skip grid nodes that are boundary condition
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

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::solveOnGridForwardEuler(Scalar dt)
{
    //explicit integration
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        for(unsigned int j = 0; j < this->particle_grid_pair_num_[i]; ++j)
        {
            const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[i][j];
            if(is_dirichlet_grid_node_(pair.node_idx_))
                continue; //skip grid nodes that are boundary condition
            Vector<Scalar,Dim> weight_gradient = pair.gradient_value_;
            SquareMatrix<Scalar,Dim> cauchy_stress = particle->cauchyStress();
            if(grid_mass_(pair.node_idx_)>std::numeric_limits<Scalar>::epsilon())
                grid_velocity_(pair.node_idx_) += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/grid_mass_(pair.node_idx_);
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::solveOnGridBackwardEuler(Scalar dt)
{
//TO DO
}

//explicit instantiations
template class MPMSolid<float,2>;
template class MPMSolid<float,3>;
template class MPMSolid<double,2>;
template class MPMSolid<double,3>;

}  //end of namespace Physika
