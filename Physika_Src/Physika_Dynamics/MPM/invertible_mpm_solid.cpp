/*
 * @file invertible_mpm_solid.cpp 
 * @brief a hybrid of FEM and CPDI2 for large deformation and invertible elasticity, uniform grid.
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
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"
#include "Physika_Dynamics/MPM/invertible_mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid()
    :CPDIMPMSolid<Scalar,Dim>(), principal_stretch_threshold_(0.3)
{
    //only works with CPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<CPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame,
                                                   Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :CPDIMPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file), principal_stretch_threshold_(0.3)
{
    //only works with CPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<CPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame,
                                                   Scalar frame_rate, Scalar max_dt, bool write_to_file,
                                                   const Grid<Scalar,Dim> &grid)
    :CPDIMPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,grid), principal_stretch_threshold_(0.3)
{
    //only works with CPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<CPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::~InvertibleMPMSolid()
{
    clearParticleDomainMesh();
}

template <typename Scalar, int Dim>
bool InvertibleMPMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::write(const std::string &file_name)
{
    //TO DO
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::read(const std::string &file_name)
{
    //TO DO
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::initSimulationData()
{
    constructParticleDomainMesh();
    resetParticleDomainData();
    MPMSolid<Scalar,Dim>::initSimulationData();
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::rasterize()
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onRasterize();
    }

    //rasterize mass and momentum of each object independently to grid (domain corner)
    //first reset data on grid and domain corners
    this->resetGridData();
    resetParticleDomainData();
    updateParticleDomainEnrichState();  //set the particle domain as enriched or not according to some criteria

    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            //determine particle type:
            //ordinary: rasterize to grid
            //transient: rasterize to grid and the enriched domain corners
            //enriched: rasterize only to domain corner
            unsigned int corner_num = (Dim==2) ? 4 : 8;
            unsigned int enriched_corner_num = 0;
            for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
            {
                unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    ++enriched_corner_num;
            }
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle get influence from grid
            {
                //rasterize to grid
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i];
                    Scalar weight = pair.weight_value_;
                    PHYSIKA_ASSERT(weight > std::numeric_limits<Scalar>::epsilon());
                    typename std::map<unsigned int,Scalar>::iterator iter = this->grid_mass_(pair.node_idx_).find(obj_idx);
                    if(iter != this->grid_mass_(pair.node_idx_).end())
                        this->grid_mass_(pair.node_idx_)[obj_idx] += weight*particle->mass();
                    else
                        this->grid_mass_(pair.node_idx_).insert(std::make_pair(obj_idx,weight*particle->mass()));
                    if(this->is_dirichlet_grid_node_(pair.node_idx_).count(obj_idx) > 0) //skip the velocity update of boundary nodes
                        continue;
                    typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator iter2 = this->grid_velocity_(pair.node_idx_).find(obj_idx);
                    if(iter2 != this->grid_velocity_(pair.node_idx_).end())
                        this->grid_velocity_(pair.node_idx_)[obj_idx] += weight*(particle->mass()*particle->velocity());
                    else
                        this->grid_velocity_(pair.node_idx_).insert(std::make_pair(obj_idx,weight*(particle->mass()*particle->velocity())));
                }
            }
            //transient/enriched particle needs to rasterize to enriched corners as well
            if(enriched_corner_num > 0)
            {
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    {
                        Scalar weight = particle_corner_weight_[obj_idx][particle_idx][corner_idx];
                        domain_corner_mass_[obj_idx][global_corner_idx] += weight*particle->mass();
                        domain_corner_velocity_[obj_idx][global_corner_idx] += weight*(particle->mass()*particle->velocity());
                    }
                } 
            }
        }
        //compute domain corner's velocity, divide momentum by mass
        for(unsigned int corner_idx = 0; corner_idx < domain_corner_mass_[obj_idx].size(); ++corner_idx)
            if(domain_corner_mass_[obj_idx][corner_idx] > std::numeric_limits<Scalar>::epsilon())
            {
                domain_corner_velocity_[obj_idx][corner_idx] /= domain_corner_mass_[obj_idx][corner_idx];
                domain_corner_velocity_before_[obj_idx][corner_idx] = domain_corner_velocity_[obj_idx][corner_idx];
            }
    }    
    //determine active grid nodes according to the grid mass of each object
    Vector<unsigned int,Dim> grid_node_num = this->grid_.nodeNum();
    std::map<unsigned int,Vector<unsigned int,Dim> > active_node_idx_1d_nd_map;
    for(typename ArrayND<std::map<unsigned int,Scalar>,Dim>::Iterator iter = this->grid_mass_.begin(); iter != this->grid_mass_.end(); ++iter)
    {
        Vector<unsigned int,Dim> node_idx = iter.elementIndex();
        for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        {
            if(this->gridMass(obj_idx,node_idx)>std::numeric_limits<Scalar>::epsilon())
            {
                unsigned int node_idx_1d = this->flatIndex(node_idx,grid_node_num);
                this->active_grid_node_.insert(std::make_pair(node_idx_1d,obj_idx));
                active_node_idx_1d_nd_map.insert(std::make_pair(node_idx_1d,node_idx));
            }
        }
    }
    //compute grid's velocity, divide momentum by mass
    for(typename std::multimap<unsigned int,unsigned int>::iterator iter = this->active_grid_node_.begin(); iter != this->active_grid_node_.end(); ++iter)
    {
        unsigned int node_idx_1d = iter->first, object_idx = iter->second;
        Vector<unsigned int,Dim> node_idx = active_node_idx_1d_nd_map[node_idx_1d];
        if(this->is_dirichlet_grid_node_(node_idx).count(object_idx) == 0) //skip grid nodes that are boundary condition
            this->grid_velocity_(node_idx)[object_idx] /= this->grid_mass_(node_idx)[object_idx];
        this->grid_velocity_before_(node_idx)[object_idx] = this->grid_velocity_(node_idx)[object_idx];  //buffer the grid velocity before any update
    }
    //if no special contact algorithm is used, multi-value at a grid node must be converted to single value for all involved objects
    if(this->contact_method_==NULL)
    {
        for(typename std::map<unsigned int,Vector<unsigned int,Dim> >::iterator iter = active_node_idx_1d_nd_map.begin();
            iter != active_node_idx_1d_nd_map.end(); ++iter)
        {
            unsigned int node_idx_1d = iter->first;
            Vector<unsigned int,Dim> node_idx = iter->second;
            if(this->active_grid_node_.count(node_idx_1d) == 1) //skip single-valued node
                continue;
            std::multimap<unsigned int,unsigned int>::iterator beg = this->active_grid_node_.lower_bound(node_idx_1d),
                end = this->active_grid_node_.upper_bound(node_idx_1d);
            std::multimap<unsigned int,unsigned int>::iterator cur = beg;
            Scalar mass_at_node = 0;
            Vector<Scalar,Dim> momentum_at_node(0);
            //accummulate values of all involved objects at this node
            while(cur != end)
            {
                mass_at_node += this->grid_mass_(node_idx)[cur->second];
                momentum_at_node += this->grid_mass_(node_idx)[cur->second] * this->grid_velocity_(node_idx)[cur->second];
                ++cur;
            }
            momentum_at_node /= mass_at_node;//velocity at node
            //set all involved objects to uniform value at this node
            for(typename std::map<unsigned int,Scalar>::iterator mass_iter = this->grid_mass_(node_idx).begin(); mass_iter != this->grid_mass_(node_idx).end(); ++mass_iter)
                mass_iter->second = mass_at_node;
            //if for any involved object, this node is set as dirichlet, then the node is dirichlet for all objects
            for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = this->grid_velocity_(node_idx).begin();
                vel_iter != this->grid_velocity_(node_idx).end(); ++vel_iter)
                if(this->is_dirichlet_grid_node_(node_idx).count(vel_iter->first))
                {
                    momentum_at_node = vel_iter->second;
                    break;
                }
            //set the velocity
            for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = this->grid_velocity_(node_idx).begin();
                vel_iter != this->grid_velocity_(node_idx).end(); ++vel_iter)
            {
                vel_iter->second = momentum_at_node;
                this->grid_velocity_before_(node_idx)[vel_iter->first] = vel_iter->second; //buffer the grid velocity before any update
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleInterpolationWeight()
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
    PHYSIKA_ASSERT(this->cpdi_update_method_);
    PHYSIKA_ASSERT(this->weight_function_);
    const GridWeightFunction<Scalar,Dim> &weight_function = *(this->weight_function_);
    //update the interpolation weight between particle and domain corner
    CPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<CPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method)  //CPDI2
        update_method->updateParticleInterpolationWeightWithEnrichment(weight_function,particle_domain_mesh_,is_enriched_domain_corner_,
                                                                       this->particle_grid_weight_and_gradient_,this->particle_grid_pair_num_,
                                                                       this->corner_grid_weight_and_gradient_,this->corner_grid_pair_num_,
                                                                       particle_corner_weight_,particle_corner_gradient_to_ref_,particle_corner_gradient_to_cur_);
    else
        PHYSIKA_ERROR("Invertible MPM only supports CPDI2!");
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleConstitutiveModelState(dt);
    }
    //update the deformation gradient with the velocity gradient from domain corners
    //the velocity of ordinary domain corners are mapped from the grid node
    unsigned int corner_num = (Dim == 2) ? 4 : 8;
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            //determine particle type
            unsigned int enriched_corner_num = 0;
            for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
            {
                unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    ++enriched_corner_num;
            }
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            SquareMatrix<Scalar,Dim> particle_vel_grad(0);
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle get influence from grid
            {
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    Vector<unsigned int,Dim> node_idx = (this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].node_idx_);
                    Vector<Scalar,Dim> weight_gradient = (this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].gradient_value_);
                    particle_vel_grad += this->gridVelocity(obj_idx,node_idx).outerProduct(weight_gradient);
                }
            }
            if(enriched_corner_num > 0) //transient/enriched particle get influence from domain corner as well
            {
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx =  particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                        particle_vel_grad += domain_corner_velocity_[obj_idx][global_corner_idx].outerProduct(particle_corner_gradient_to_cur_[obj_idx][particle_idx][corner_idx]);
                }
            }
            SquareMatrix<Scalar,Dim> particle_deform_grad = particle->deformationGradient();
            //use the remedy in <Augmented MPM for phase-change and varied materials> to prevent |F| < 0
            SquareMatrix<Scalar,Dim> identity = SquareMatrix<Scalar,Dim>::identityMatrix(); 
            if((identity + dt*particle_vel_grad).determinant() > 0) //normal update
                particle_deform_grad += dt*particle_vel_grad*particle_deform_grad;
            else //the remedy
                particle_deform_grad += (dt*particle_vel_grad + 0.25*dt*dt*particle_vel_grad*particle_vel_grad)*particle_deform_grad;
            PHYSIKA_ASSERT(particle_deform_grad.determinant() > 0);
            particle->setDeformationGradient(particle_deform_grad);  //update deformation gradient
            Scalar particle_vol = (particle_deform_grad.determinant())*(this->particle_initial_volume_[obj_idx][particle_idx]);
            particle->setVolume(particle_vol);  //update particle volume
        }
    }
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleVelocity()
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleVelocity();
    }
    //interpolate delta of grid/corner velocity to particle
    //some are interpolated from grid, some are from domain corner
    unsigned int corner_num = (Dim == 2) ? 4 : 8;
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            if(this->is_dirichlet_particle_[obj_idx][particle_idx])
                continue;//skip boundary particles
            //determine particle type
            unsigned int enriched_corner_num = 0;
            for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
            {
                unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    ++enriched_corner_num;
            }
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            Vector<Scalar,Dim> new_vel = particle->velocity();
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle get influence from grid
            {
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    Vector<unsigned int,Dim> node_idx = (this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].node_idx_);
                    if(this->gridMass(obj_idx,node_idx) <= std::numeric_limits<Scalar>::epsilon())
                        continue;
                    Scalar weight = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].weight_value_;
                    Vector<Scalar,Dim> cur_grid_vel(0),grid_vel_before(0);
                    if(this->grid_velocity_(node_idx).find(obj_idx) != this->grid_velocity_(node_idx).end())
                        cur_grid_vel = this->grid_velocity_(node_idx)[obj_idx];
                    else
                        PHYSIKA_ERROR("Error in updateParticleVelocity!");
                    if(this->grid_velocity_before_(node_idx).find(obj_idx) != this->grid_velocity_before_(node_idx).end())
                        grid_vel_before = this->grid_velocity_before_(node_idx)[obj_idx];
                    else
                        PHYSIKA_ERROR("Error in updateParticleVelocity!");
                    new_vel += weight*(cur_grid_vel-grid_vel_before);
                }
            }
            if(enriched_corner_num > 0) //transient/enriched particle get influence from domain corner as well
            {
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx =  particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    {
                        Scalar weight = particle_corner_weight_[obj_idx][obj_idx][corner_idx];
                        new_vel += weight*(domain_corner_velocity_[obj_idx][global_corner_idx]-domain_corner_velocity_before_[obj_idx][global_corner_idx]);
                    }
                }
            }
            particle->setVelocity(new_vel);
        }
    }
}
 
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticlePosition(Scalar dt)
{
    CPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<CPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method)  //CPDI2
    {
        //plugin operation
        MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
        for(unsigned int i = 0; i < this->plugins_.size(); ++i)
        {
            plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
            if(plugin)
                plugin->onUpdateParticlePosition(dt);
        }
        //update particle domain before update particle position
        //some domain corners(ordinary) are updated with the velocity from grid, some(enriched) are updated with the corner velocity
        unsigned int corner_num = (Dim == 2) ? 4 : 8;
        for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        {
            for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
            {
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    Vector<Scalar,Dim> new_corner_pos = this->particle_domain_corners_[obj_idx][particle_idx][corner_idx];
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx]) //update with corner's velocity
                        new_corner_pos += domain_corner_velocity_[obj_idx][global_corner_idx]*dt;
                    else  //update with velocity from grid
                    {
                        for(unsigned int i = 0; i < this->corner_grid_pair_num_[obj_idx][particle_idx][corner_idx]; ++i)
                        {
                            const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->corner_grid_weight_and_gradient_[obj_idx][particle_idx][corner_idx][i];
                            Scalar weight = pair.weight_value_;
                            Vector<Scalar,Dim> node_vel = this->gridVelocity(obj_idx,pair.node_idx_);
                            new_corner_pos += weight*node_vel*dt;
                        }
                    }
                    this->particle_domain_corners_[obj_idx][particle_idx][corner_idx] = new_corner_pos;
                    particle_domain_mesh_[obj_idx]->setVertPos(global_corner_idx,new_corner_pos);
                }
            }
        }
        //update particle position with CPDI2
        update_method->updateParticlePosition(dt,this->is_dirichlet_particle_);
    }
    else
        PHYSIKA_ERROR("Invertible MPM only supports CPDI2!");
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::setCurrentParticleDomain(unsigned int object_idx, unsigned int particle_idx,
                                                              const ArrayND<Vector<Scalar,Dim>,Dim> &particle_domain_corner)
{
    CPDIMPMSolid<Scalar,Dim>::setCurrentParticleDomain(object_idx,particle_idx,particle_domain_corner);
    //set the data in particle_domain_mesh_ as well
    unsigned int corner_idx = 0;
    for(typename ArrayND<Vector<Scalar,Dim>,Dim>::ConstIterator iter = particle_domain_corner.begin(); iter != particle_domain_corner.end(); ++corner_idx,++iter)
    {
        PHYSIKA_ASSERT(particle_domain_mesh_[object_idx]);
        particle_domain_mesh_[object_idx]->setEleVertPos(particle_idx,corner_idx,*iter);
    }
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::setPrincipalStretchThreshold(Scalar threshold)
{
    if(threshold < 0)
    {
        std::cerr<<"Warning: invalid principal threshold provided, default value (0.3) is used instead!\n";
        principal_stretch_threshold_ = 0.3;
    }
    else
        principal_stretch_threshold_ = threshold;
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridForwardEuler(Scalar dt)
{
    //explicit integration
    //integration on grid and domain corner
    unsigned int corner_num = (Dim == 2) ? 4 : 8;
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            //determine particle type
            unsigned int enriched_corner_num = 0;
            for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
            {
                unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    ++enriched_corner_num;
            }
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle get influence from grid
            {
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i];
                    if(this->is_dirichlet_grid_node_(pair.node_idx_).count(obj_idx) > 0)
                        continue; //skip grid nodes that are boundary condition
                    Vector<Scalar,Dim> weight_gradient = pair.gradient_value_;
                    SquareMatrix<Scalar,Dim> cauchy_stress = particle->cauchyStress();
                    if(this->grid_mass_(pair.node_idx_)[obj_idx] <= std::numeric_limits<Scalar>::epsilon())
                        continue; //skip grid nodes with near zero mass
                    if(this->contact_method_)  //if contact method other than the inherent one is employed, update the grid velocity of each object independently
                        this->grid_velocity_(pair.node_idx_)[obj_idx] += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/this->grid_mass_(pair.node_idx_)[obj_idx];
                    else  //otherwise, grid velocity of all objects that ocuppy the node get updated
                    {
                        if(this->is_dirichlet_grid_node_(pair.node_idx_).size() > 0)
                            continue;  //if for any involved object, this node is set as dirichlet, then the node is dirichlet for all objects
                        for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = this->grid_velocity_(pair.node_idx_).begin();
                            vel_iter != this->grid_velocity_(pair.node_idx_).end(); ++vel_iter)
                            if(this->gridMass(vel_iter->first,pair.node_idx_) > std::numeric_limits<Scalar>::epsilon())
                                vel_iter->second += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/this->grid_mass_(pair.node_idx_)[obj_idx];
                    }
                }
            }
            if(enriched_corner_num > 0) //transient/enriched particle get influence from domain corner as well
            {
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx =  particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    {
                        if(domain_corner_mass_[obj_idx][global_corner_idx] <= std::numeric_limits<Scalar>::epsilon())
                            continue;
                        Vector<Scalar,Dim> weight_gradient = particle_corner_gradient_to_ref_[obj_idx][particle_idx][corner_idx]; //weigt gradient with respect to reference configuration
                        SquareMatrix<Scalar,Dim> deform_grad = particle->deformationGradient();
                        SquareMatrix<Scalar,Dim> left_rotation, diag_deform_grad, right_rotation;
                        diagonalizeDeformationGradient(deform_grad,left_rotation,diag_deform_grad,right_rotation);
                        //clamp the principal stretch to the threshold if it's compressed too severely
                        for(unsigned int row = 0; row < Dim; ++row)
                            if(diag_deform_grad(row,row) < principal_stretch_threshold_)
                                diag_deform_grad(row,row) = principal_stretch_threshold_;
                        //temporarily set the deformation gradient of the particle to the diagonalized one to compute the unrotated stress
                        particle->setDeformationGradient(diag_deform_grad);
                        // P = U*P^*V^T
                        SquareMatrix<Scalar,Dim> first_PiolaKirchoff_stress = left_rotation*(particle->firstPiolaKirchhoffStress())*(right_rotation.transpose());
                        //recover the deformation gradient of the particle
                        particle->setDeformationGradient(deform_grad);
                        //compute internal force with P and reference configuration
                        Scalar particle_initial_volume = this->particle_initial_volume_[obj_idx][particle_idx];
                        domain_corner_velocity_[obj_idx][global_corner_idx] += dt*(-1)*particle_initial_volume*first_PiolaKirchoff_stress*weight_gradient/domain_corner_mass_[obj_idx][global_corner_idx];

                        // Vector<Scalar,Dim> weight_gradient = particle_corner_gradient_to_cur_[obj_idx][particle_idx][corner_idx];
                        // SquareMatrix<Scalar,Dim> cauchy_stress = particle->cauchyStress();
                        // domain_corner_velocity_[obj_idx][global_corner_idx] += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/domain_corner_mass_[obj_idx][global_corner_idx];
                    }
                }
            }
        }
    }
    //apply gravity
    applyGravityOnEnrichedDomainCorner(dt);    
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridBackwardEuler(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject()
{
    CPDIMPMSolid<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject();
    unsigned int corner_num = Dim==2 ? 4 : 8;
    unsigned int last_object_idx = this->objectNum() - 1;
    unsigned int particle_num_of_last_object = this->particleNumOfObject(last_object_idx);
    std::vector<Scalar> one_particle_corner_weight(corner_num,0);
    std::vector<Vector<Scalar,Dim> > one_particle_corner_gradient(corner_num,Vector<Scalar,Dim>(0));
    std::vector<std::vector<Scalar> > all_particle_corner_weight(particle_num_of_last_object,one_particle_corner_weight);
    std::vector<std::vector<Vector<Scalar,Dim> > > all_particle_corner_gradient(particle_num_of_last_object,one_particle_corner_gradient);
    particle_corner_weight_.push_back(all_particle_corner_weight);
    particle_corner_gradient_to_ref_.push_back(all_particle_corner_gradient);
    particle_corner_gradient_to_cur_.push_back(all_particle_corner_gradient);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(unsigned int object_idx)
{
    CPDIMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(object_idx);
    unsigned int corner_num = Dim==2 ? 4 : 8;
    std::vector<Scalar> one_particle_corner_weight(corner_num,0);
    std::vector<Vector<Scalar,Dim> > one_particle_corner_gradient(corner_num,Vector<Scalar,Dim>(0));
    particle_corner_weight_[object_idx].push_back(one_particle_corner_weight);
    particle_corner_gradient_to_ref_[object_idx].push_back(one_particle_corner_gradient);
    particle_corner_gradient_to_cur_[object_idx].push_back(one_particle_corner_gradient);
}
 
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(unsigned int object_idx)
{
    CPDIMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(object_idx);
    typename std::vector<std::vector<std::vector<Scalar> > >::iterator iter1 = particle_corner_weight_.begin() + object_idx;
    particle_corner_weight_.erase(iter1);
    typename std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > >::iterator iter2 = particle_corner_gradient_to_ref_.begin() + object_idx;
    particle_corner_gradient_to_ref_.erase(iter2);
    typename std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > >::iterator iter3 = particle_corner_gradient_to_cur_.begin() + object_idx;
    particle_corner_gradient_to_cur_.erase(iter3);
}
 
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx)
{
    CPDIMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(object_idx,particle_idx);
    typename std::vector<std::vector<Scalar> >::iterator iter1 = particle_corner_weight_[object_idx].begin() + particle_idx;
    particle_corner_weight_[object_idx].erase(iter1);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter2 = particle_corner_gradient_to_ref_[object_idx].begin() + particle_idx;
    particle_corner_gradient_to_ref_[object_idx].erase(iter2);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter3 = particle_corner_gradient_to_cur_[object_idx].begin() + particle_idx;
    particle_corner_gradient_to_cur_[object_idx].erase(iter3);
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::resetParticleDomainData()
{
    Vector<Scalar,Dim> zero_vel(0);
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        unsigned int domain_corner_num = particle_domain_mesh_[obj_idx]->vertNum();
        for(unsigned int domain_corner_idx = 0; domain_corner_idx < domain_corner_num; ++domain_corner_idx)
        {
            is_enriched_domain_corner_[obj_idx][domain_corner_idx] = 0x00;
            domain_corner_mass_[obj_idx][domain_corner_idx] = 0;
            domain_corner_velocity_[obj_idx][domain_corner_idx] = zero_vel;
            domain_corner_velocity_before_[obj_idx][domain_corner_idx] = zero_vel;
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::constructParticleDomainMesh()
{
    clearParticleDomainMesh(); //clear potential space
    //resize 
    unsigned int obj_num = this->objectNum();
    particle_domain_mesh_.resize(obj_num);
    is_enriched_domain_corner_.resize(obj_num);
    domain_corner_mass_.resize(obj_num);
    domain_corner_velocity_.resize(obj_num);
    domain_corner_velocity_before_.resize(obj_num);
    //construct mesh
    std::vector<Vector<Scalar,Dim> > corner_positions;
    unsigned int *domains = NULL;
    unsigned int corner_num = (Dim==2)?4:8;
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        unsigned int particle_num = this->particleNumOfObject(obj_idx);
        domains = new unsigned int[particle_num*corner_num];
        corner_positions.clear();
        for(unsigned int particle_idx = 0; particle_idx < particle_num; ++particle_idx)
        {
            for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
            {
                typename std::vector<Vector<Scalar,Dim> >::iterator iter = std::find(corner_positions.begin(),corner_positions.end(),this->particle_domain_corners_[obj_idx][particle_idx][corner_idx]);
                if(iter == corner_positions.end())
                {
                    corner_positions.push_back(this->particle_domain_corners_[obj_idx][particle_idx][corner_idx]);
                    domains[particle_idx*corner_num+corner_idx] = corner_positions.size() - 1;
                }
                else
                {
                    unsigned int idx = static_cast<unsigned int>(iter-corner_positions.begin());
                    domains[particle_idx*corner_num+corner_idx] = idx;
                }
            }
        }
        unsigned int vert_num = corner_positions.size();
        Scalar *vertices = new Scalar[vert_num*Dim];
        for(unsigned int i = 0; i < corner_positions.size(); ++i)
            for(unsigned int j =0; j < Dim; ++j)
                vertices[i*Dim+j] = corner_positions[i][j];
        if(Dim == 2)
            particle_domain_mesh_[obj_idx] = dynamic_cast<VolumetricMesh<Scalar,Dim>*>(new QuadMesh<Scalar>(vert_num,vertices,particle_num,domains));
        else
            particle_domain_mesh_[obj_idx] = dynamic_cast<VolumetricMesh<Scalar,Dim>*>(new CubicMesh<Scalar>(vert_num,vertices,particle_num,domains));
        delete[] domains;
        delete[] vertices;
        //resize
        is_enriched_domain_corner_[obj_idx].resize(vert_num);
        domain_corner_mass_[obj_idx].resize(vert_num);
        domain_corner_velocity_[obj_idx].resize(vert_num);
        domain_corner_velocity_before_[obj_idx].resize(vert_num);
    }
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::clearParticleDomainMesh()
{
    for(unsigned int i = 0; i < particle_domain_mesh_.size(); ++i)
        if(particle_domain_mesh_[i])
            delete particle_domain_mesh_[i];
}
    
template <typename Scalar, int Dim>
bool InvertibleMPMSolid<Scalar,Dim>::isEnrichCriteriaSatisfied(unsigned int obj_idx, unsigned int particle_idx) const
{
    // unsigned int obj_num = this->objectNum();
    // PHYSIKA_ASSERT(obj_idx<obj_num);
    // unsigned int particle_num = this->particleNumOfObject(obj_idx);
    // PHYSIKA_ASSERT(particle_idx<particle_num);
    // //rule one: if there's any dirichlet grid node within the range of the particle, the particle cannot be enriched
    // //the dirichlet boundary is correctly enforced in this way
    // for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
    // {
    //     Vector<unsigned int,Dim> node_idx = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].node_idx_;
    //     if(this->is_dirichlet_grid_node_(node_idx).count(obj_idx) > 0)
    //         return false;
    // }
    // // //rule two: only enrich while compression
    // // if(this->particles_[obj_idx][particle_idx]->volume() > this->particle_initial_volume_[obj_idx][particle_idx])
    // //     return false;
    return true;
    //TO DO
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleDomainEnrichState()
{
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            if(isEnrichCriteriaSatisfied(obj_idx,particle_idx))
            {
                PHYSIKA_ASSERT(particle_domain_mesh_[obj_idx]);
                unsigned int corner_num = particle_domain_mesh_[obj_idx]->eleVertNum(particle_idx);
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    is_enriched_domain_corner_[obj_idx][global_corner_idx] = 0x01;
                }
            }
        }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::applyGravityOnEnrichedDomainCorner(Scalar dt)
{
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        for(unsigned int corner_idx = 0; corner_idx < domain_corner_velocity_[obj_idx].size(); ++corner_idx)
            if(is_enriched_domain_corner_[obj_idx][corner_idx])
                domain_corner_velocity_[obj_idx][corner_idx][1] += dt*(-1)*(this->gravity_);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::diagonalizeDeformationGradient(const SquareMatrix<Scalar,2> &deform_grad,
                                                                    SquareMatrix<Scalar,2> &left_rotation,
                                                                    SquareMatrix<Scalar,2> &diag_deform_grad,
                                                                    SquareMatrix<Scalar,2> &right_rotation) const
{
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon(); //the epsilon between the entries and zero
    //naming correspondence with that in paper: U, left_rotation; V, right_rotation
    SquareMatrix<Scalar,2> F_transpose_F = deform_grad.transpose()*deform_grad;
    //imag part is dummy because F^T*F is symmetric matrix
    Vector<Scalar,2> eigen_values_real, eigen_values_imag;
    SquareMatrix<Scalar,2> eigen_vectors_real, eigen_vectors_imag;
    F_transpose_F.eigenDecomposition(eigen_values_real,eigen_values_imag,eigen_vectors_real,eigen_vectors_imag);
    right_rotation = eigen_vectors_real; //V is right rotation
    //Case I, det(V) == -1: simply multiply a column of V by -1
    if(right_rotation.determinant() < 0)
    {
        //here we negate the first column
        right_rotation(0,0) *= -1;
        right_rotation(1,0) *= -1;
    }
    //diagonal entries
    diag_deform_grad(0,0) = eigen_values_real[0] > 0 ? sqrt(eigen_values_real[0]) : 0;
    diag_deform_grad(1,1) = eigen_values_real[1] > 0 ? sqrt(eigen_values_real[1]) : 0;
    diag_deform_grad(0,1) = diag_deform_grad(1,0) = 0;
    //inverse of F^
    SquareMatrix<Scalar,2> diag_deform_grad_inverse(0);
    diag_deform_grad_inverse(0,0) = diag_deform_grad(0,0) > epsilon ? 1/diag_deform_grad(0,0) : 0;
    diag_deform_grad_inverse(1,1) = diag_deform_grad(1,1) > epsilon ? 1/diag_deform_grad(1,1) : 0;
    // U = F*V*inv(F^)
    left_rotation = deform_grad * right_rotation;
    left_rotation *= diag_deform_grad_inverse;
    //Case II, an entry of F^ is near zero
    //set the corresponding column of U to be orthonormal to other columns
    if(diag_deform_grad(0,0) < epsilon && diag_deform_grad(1,1) < epsilon)
    {
        //extreme case: material has collapsed almost to a point
        //U = I
        left_rotation = SquareMatrix<Scalar,2>::identityMatrix();
    }
    else
    {
        bool done = false;
        for(unsigned int col = 0; col < 2; ++col)
        {
            unsigned int col_a = col, col_b = (col+1)%2;
            if(diag_deform_grad(col_a,col_a) < epsilon)
            {
                //entry a of F^ is near zero, set column a of U to be orthonormal with  column b
                left_rotation(0,col_a) = left_rotation(1,col_b);
                left_rotation(1,col_a) = -left_rotation(0,col_b);
                //the orthonormal vector leads to |U|<0, need to negate it
                if(left_rotation.determinant() < 0)
                {
                    left_rotation(0,col_a) *= -1;
                    left_rotation(1,col_a) *= -1;
                }
                done = true;
                break;
            }
        }
        if(!done)
        {
            //Case III, det(U) = -1: negate the minimal element of F^ and corresponding column of U
            if(left_rotation.determinant() < 0)
            {
                unsigned int smallest_value_idx = (diag_deform_grad(0,0) < diag_deform_grad(1,1)) ? 0 : 1;
                //negate smallest singular value
                diag_deform_grad(smallest_value_idx,smallest_value_idx) *= -1;
                left_rotation(0,smallest_value_idx) *= -1;
                left_rotation(1,smallest_value_idx) *= -1;
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::diagonalizeDeformationGradient(const SquareMatrix<Scalar,3> &deform_grad,
                                                                    SquareMatrix<Scalar,3> &left_rotation,
                                                                    SquareMatrix<Scalar,3> &diag_deform_grad,
                                                                    SquareMatrix<Scalar,3> &right_rotation) const
{
    Scalar epsilon = std::numeric_limits<Scalar>::epsilon(); //the epsilon between the entries and zero
    //naming correspondence with that in paper: U, left_rotation; V, right_rotation
    SquareMatrix<Scalar,3> F_transpose_F = deform_grad.transpose()*deform_grad;
    //imag part is dummy because F^T*F is symmetric matrix
    Vector<Scalar,3> eigen_values_real, eigen_values_imag;
    SquareMatrix<Scalar,3> eigen_vectors_real, eigen_vectors_imag;
    F_transpose_F.eigenDecomposition(eigen_values_real,eigen_values_imag,eigen_vectors_real,eigen_vectors_imag);
    right_rotation = eigen_vectors_real; //V is right rotation
    //Case I, det(V) == -1: simply multiply a column of V by -1
    if(right_rotation.determinant() < 0)
    {
        //here we negate the first column
        for(unsigned int row = 0; row < 3; ++row)
            right_rotation(row,0) *= -1;
    }
    //diagonal entries
    for(unsigned int row = 0; row < 3; ++row)
        for(unsigned int col = 0; col < 3; ++col)
            if(row == col)
                diag_deform_grad(row,col) = eigen_values_real[row] > 0 ? sqrt(eigen_values_real[row]) : 0;
            else
                diag_deform_grad(row,col) = 0;
    //inverse of F^
    SquareMatrix<Scalar,3> diag_deform_grad_inverse(0);
    for(unsigned int row = 0; row < 3; ++row)
        diag_deform_grad_inverse(row,row) = diag_deform_grad(row,row) > epsilon ? 1/diag_deform_grad(row,row) : 0;
    // U = F*V*inv(F^)
    left_rotation = deform_grad * right_rotation;
    left_rotation *= diag_deform_grad_inverse;
    //Case II, an entry of F^ is near zero
    //set the corresponding column of U to be orthonormal to other columns
    bool extreme_case = true; // F = 0
    for(unsigned int row = 0; row < 3; ++row)
    {
        extreme_case =  extreme_case && (diag_deform_grad(row,row) < epsilon);
        if(extreme_case == false)
            break;
    }
    if(extreme_case)
    {
        //extreme case: material has collapsed almost to a point
        //U = I
        left_rotation = SquareMatrix<Scalar,3>::identityMatrix();
    }
    else
    {
        bool done = false;
        for(unsigned int col = 0; col < 3; ++col)
        {
            unsigned int col_a = col, col_b = (col+1)%3, col_c = (col+2)%3;
            if(diag_deform_grad(col_b,col_b) < epsilon && diag_deform_grad(col_c,col_c) < epsilon)
            {
                //two entries of F^ are near zero: set corresponding columns of U to be orthonormal to the remainning one
                Vector<Scalar,3> left_rotation_col_a;
                for(unsigned int row = 0; row < 3; ++row)
                    left_rotation_col_a[row] = left_rotation(row,col_a);
                Vector<Scalar,3> left_rotation_col_b; //b is chosen to be orthogonal to a
                unsigned int smallest_idx = 0;
                for(unsigned int row = 1; row < 3; ++row)
                    if(abs(left_rotation_col_a[row]) < abs(left_rotation_col_a[smallest_idx]))
                        smallest_idx = row;
                Vector<Scalar,3> axis(0);
                axis[smallest_idx] = 1;
                left_rotation_col_b = left_rotation_col_a.cross(axis);
                left_rotation_col_b.normalize();
                Vector<Scalar,3> left_rotation_col_c = left_rotation_col_a.cross(left_rotation_col_b);
                left_rotation_col_c.normalize();
                for(unsigned int row = 0; row < 3; ++row)
                {
                    left_rotation(row,col_b) = left_rotation_col_b[row];
                    left_rotation(row,col_c) = left_rotation_col_c[row];
                }
                //the orthonormal vector leads to |U|<0, need to negate it
                if(left_rotation.determinant() < 0)
                {
                    for(unsigned int row = 0; row < 3; ++row)
                        left_rotation(row,col_b) *= -1;
                }
                done = true;
                break;
            }
        }
        if(!done)
        {
            for(unsigned int col = 0; col < 3; ++col)
            {
                unsigned int col_a = col, col_b = (col+1)%3, col_c = (col+2)%3;
                if(diag_deform_grad(col_a,col_a) < epsilon)
                {
                    //only one entry of F^ is near zero
                    Vector<Scalar,3> left_rotation_col_b, left_rotation_col_c;
                    for(unsigned int row = 0; row < 3; ++row)
                    {
                        left_rotation_col_b[row] = left_rotation(row,col_b);
                        left_rotation_col_c[row] = left_rotation(row,col_c);
                    }
                    Vector<Scalar,3> left_rotation_col_a = left_rotation_col_b.cross(left_rotation_col_c);
                    left_rotation_col_a.normalize();
                    for(unsigned int row = 0; row < 3; ++row)
                        left_rotation(row,col_a) = left_rotation_col_a[row];
                    //the orthonormal vector leads to |U|<0, need to negate it
                    if(left_rotation.determinant() < 0)
                    {
                        for(unsigned int row = 0; row < 3; ++row)
                            left_rotation(row,col_a) *= -1;
                    }
                    done = true;
                    break;
                }
            }
        }
        if(!done)
        {
            //Case III, det(U) = -1: negate the minimal element of F^ and corresponding column of U
            if(left_rotation.determinant() < 0)
            {
                unsigned int smallest_value_idx = 0;
                for(unsigned int i = 1; i < 3; ++i)
                    if(diag_deform_grad(i,i) < diag_deform_grad(smallest_value_idx,smallest_value_idx))
                        smallest_value_idx = i;
                //negate smallest singular value
                diag_deform_grad(smallest_value_idx,smallest_value_idx) *= -1;
                for(unsigned int i = 0; i < 3; ++i)
                    left_rotation(i,smallest_value_idx) *= -1;
            }
        }
    }
}
 
//explicit instantiation
template class InvertibleMPMSolid<float,2>;
template class InvertibleMPMSolid<double,2>;
template class InvertibleMPMSolid<float,3>;
template class InvertibleMPMSolid<double,3>;

}  //end of namespace Physika
