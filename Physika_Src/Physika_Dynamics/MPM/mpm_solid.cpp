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

#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Driver/driver_plugin_base.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/MPM_Step_Methods/mpm_step_method.h"
#include "Physika_Dynamics/MPM/Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
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
void MPMSolid<Scalar,Dim>::addPlugin(DriverPluginBase<Scalar> *plugin)
{
//TO DO
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
void MPMSolid<Scalar,Dim>::rasterize()
{
    resetGridData();
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> weight_support_radius, grid_dx = (this->grid_).dX();
    for(unsigned int i = 0; i < Dim; ++i)
        weight_support_radius[i] = grid_dx[i]*(this->weight_radius_cell_scale_[i]);
    //rasterize mass and momentum to grid
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        for(InfluenceIterator iter(this->grid_,particle_pos,this->weight_radius_cell_scale_); iter.valid(); ++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
            Scalar weight = this->weight_function_->weight(particle_to_node,weight_support_radius); 
            grid_mass_(node_idx) += weight*particle->mass();
            grid_velocity_(node_idx) += weight*(particle->mass()*particle->velocity());
        }
    }
    //determine active grid nodes according to the grid mass
    for(typename ArrayND<Scalar,Dim>::Iterator iter = grid_mass_.begin(); iter != grid_mass_.end(); ++iter)
    {
        Vector<unsigned int,Dim> ele_idx = iter.elementIndex();
        if(grid_mass_(ele_idx)>0)
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
    //only explicit integration is implemented yet
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> weight_support_radius, grid_dx = (this->grid_).dX();
    for(unsigned int i = 0; i < Dim; ++i)
        weight_support_radius[i] = grid_dx[i]*(this->weight_radius_cell_scale_[i]);
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        for(InfluenceIterator iter(this->grid_,particle_pos,this->weight_radius_cell_scale_); iter.valid(); ++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
            Vector<Scalar,Dim> weight_gradient = this->weight_function_->gradient(particle_to_node,weight_support_radius);
            SquareMatrix<Scalar,Dim> cauchy_stress = particle->cauchyStress();
            grid_velocity_(node_idx) += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/grid_mass_(node_idx);
        }
    }
    //TO DO (implicit)
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::performGridCollision(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::performParticleCollision(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticleInterpolationWeight()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> weight_support_radius, grid_dx = (this->grid_).dX();
    for(unsigned int i = 0; i < Dim; ++i)
        weight_support_radius[i] = grid_dx[i]*(this->weight_radius_cell_scale_[i]);
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        SquareMatrix<Scalar,Dim> particle_vel_grad(0);
        for(InfluenceIterator iter(this->grid_,particle_pos,this->weight_radius_cell_scale_); iter.valid(); ++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
            Vector<Scalar,Dim> weight_gradient = this->weight_function_->gradient(particle_to_node,weight_support_radius);
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
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> weight_support_radius, grid_dx = (this->grid_).dX();
    for(unsigned int i = 0; i < Dim; ++i)
        weight_support_radius[i] = grid_dx[i]*(this->weight_radius_cell_scale_[i]);
    //direct interpolate grid velocity to particle
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> particle_pos = particle->position();
        Vector<Scalar,Dim> new_vel(0);
        for(InfluenceIterator iter(this->grid_,particle_pos,this->weight_radius_cell_scale_); iter.valid(); ++iter)
        {
            Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
            Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
            Scalar weight = this->weight_function_->weight(particle_to_node,weight_support_radius); 
            new_vel += weight*grid_velocity_(node_idx);
        }
        particle->setVelocity(new_vel);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::updateParticlePosition(Scalar dt)
{
    //update particle's position with the new velocity
    for(unsigned int i = 0; i < this->particles_.size(); ++i)
    {
        SolidParticle<Scalar,Dim> *particle = this->particles_[i];
        Vector<Scalar,Dim> new_pos = particle->position() + particle->velocity()*dt;
        particle->setPosition(new_pos);
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::initialize()
{
//TO DO
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

//explicit instantiations
template class MPMSolid<float,2>;
template class MPMSolid<float,3>;
template class MPMSolid<double,2>;
template class MPMSolid<double,3>;

}  //end of namespace Physika
