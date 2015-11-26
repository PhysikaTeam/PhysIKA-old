/*
 * @file invertible_mpm_solid.cpp
 * @brief a hybrid of FEM and modified CPDI2 for large deformation and invertible elasticity, uniform grid.
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
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
#include <map>
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Numerics/Linear_System_Solvers/conjugate_gradient_solver.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/Collidable_Objects/collidable_object.h"
#include "Physika_Dynamics/Utilities/Grid_Weight_Function_Influence_Iterators/uniform_grid_weight_function_influence_iterator.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/robust_CPDI2_update_method.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"
#include "Physika_Dynamics/MPM/MPM_Linear_Systems/invertible_mpm_solid_linear_system.h"
#include "Physika_Dynamics/MPM/invertible_mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolid<Scalar,Dim>::default_enrich_metric_ = 0.6;

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid()
    :CPDIMPMSolid<Scalar,Dim>(), principal_stretch_threshold_(0.1),
     enable_enrichment_(true), enable_entire_enrichment_(false),
     invertible_mpm_system_(NULL), invertible_system_rhs_(NULL), invertible_system_x_(NULL)
{
    //only works with RobustCPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<RobustCPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame,
                                                   Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :CPDIMPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file), principal_stretch_threshold_(0.1),
    enable_enrichment_(true), enable_entire_enrichment_(false),
    invertible_mpm_system_(NULL), invertible_system_rhs_(NULL), invertible_system_x_(NULL)
{
    //only works with RobustCPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<RobustCPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame,
                                                   Scalar frame_rate, Scalar max_dt, bool write_to_file,
                                                   const Grid<Scalar,Dim> &grid)
    :CPDIMPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,grid), principal_stretch_threshold_(0.1),
    enable_enrichment_(true), enable_entire_enrichment_(false),
    invertible_mpm_system_(NULL), invertible_system_rhs_(NULL), invertible_system_x_(NULL)
{
    //only works with RobustCPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<RobustCPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::~InvertibleMPMSolid()
{
    clearParticleDomainMesh();
    if (invertible_mpm_system_)
        delete invertible_mpm_system_;
    if (invertible_system_rhs_)
        delete invertible_system_rhs_;
    if (invertible_system_x_)
        delete invertible_system_x_;
}

template <typename Scalar, int Dim>
bool InvertibleMPMSolid<Scalar,Dim>::withRestartSupport() const
{
    return false;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::write(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::read(const std::string &file_name)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolid<Scalar,Dim>::computeTimeStep()
{
    Scalar min_cell_size = this->minCellEdgeLength();
    Scalar max_particle_vel = this->maxParticleVelocityNorm();
    Scalar max_corner_vel = (std::numeric_limits<Scalar>::min)();
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        for(unsigned int corner_idx = 0; corner_idx < particle_domain_mesh_[obj_idx]->vertNum(); ++corner_idx)
        {
            if(is_enriched_domain_corner_[obj_idx][corner_idx])
            {
                Scalar corner_vel_sqr = domain_corner_velocity_[obj_idx][corner_idx].normSquared();
                max_corner_vel = max_corner_vel > corner_vel_sqr ? max_corner_vel : corner_vel_sqr;
            }
        }
    max_corner_vel = sqrt(max_corner_vel);
    Scalar max_vel = max(max_particle_vel,max_corner_vel);
    this->dt_ = (this->cfl_num_ * min_cell_size)/(this->sound_speed_+max_vel);
    this->dt_ = this->dt_ > this->max_dt_ ? this->max_dt_ : this->dt_;
    return this->dt_;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::initSimulationData()
{
    constructParticleDomainMesh();
    resetParticleDomainData();
    computeParticleInterpolationWeightAndGradientInInitialDomain(); //precomputation, in reference configuration
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
            unsigned int enriched_corner_num = enrichedDomainCornerNum(obj_idx,particle_idx);
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle get influence from grid
            {
                //rasterize to grid
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i];
                    Scalar weight = pair.weight_value;
                    PHYSIKA_ASSERT(weight > std::numeric_limits<Scalar>::epsilon());
                    typename std::map<unsigned int,Scalar>::iterator iter = this->grid_mass_(pair.node_idx).find(obj_idx);
                    if(iter != this->grid_mass_(pair.node_idx).end())
                        this->grid_mass_(pair.node_idx)[obj_idx] += weight*particle->mass();
                    else
                        this->grid_mass_(pair.node_idx).insert(std::make_pair(obj_idx,weight*particle->mass()));
                    if(this->is_dirichlet_grid_node_(pair.node_idx).count(obj_idx) > 0) //skip the velocity update of boundary nodes
                        continue;
                    typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator iter2 = this->grid_velocity_(pair.node_idx).find(obj_idx);
                    if(iter2 != this->grid_velocity_(pair.node_idx).end())
                        this->grid_velocity_(pair.node_idx)[obj_idx] += weight*(particle->mass()*particle->velocity());
                    else
                        this->grid_velocity_(pair.node_idx).insert(std::make_pair(obj_idx,weight*(particle->mass()*particle->velocity())));
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
void InvertibleMPMSolid<Scalar,Dim>::resolveContactOnParticles(Scalar dt)
{
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onResolveContactOnParticles(dt);
    }
    //resolve contact with the kinematic objects in scene on the particle level
    //detect collision using the domain corners, mark the domain corner that is in contact as colliding
    //impulse on the domain corners will be computed and  interpolated to particles
    //the colliding corners will be projected onto the surface of colliding object
    if(!(this->collidable_objects_).empty())
    {
        for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
            for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
            {
                //resolve contact to colliding corners
                unsigned int corner_num = Dim == 2 ? 4 : 8;
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    Vector<Scalar,Dim> corner_pos = this->particle_domain_corners_[obj_idx][particle_idx][corner_idx];
                    unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    //skip the corner that is in a cell with all dirichlet nodes
                    const Grid<Scalar,Dim> &grid = this->grid();
                    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
                    Vector<unsigned int,Dim> cell_idx;
                    Vector<Scalar,Dim> bias_in_cell;
                    grid.cellIndexAndBiasInCell(corner_pos,cell_idx,bias_in_cell);
                    Vector<unsigned int,Dim> node_idx;
                    unsigned int cell_node_num = Dim == 2 ? 4 : 8;
                    Vector<unsigned int,Dim> cell_node_dim(2);
                    unsigned int cell_dirichlet_node_num = 0;
                    for(unsigned int flat_idx = 0; flat_idx < cell_node_num; ++flat_idx)
                    {
                        node_idx = cell_idx + this->multiDimIndex(flat_idx,cell_node_dim);
                        if(this->is_dirichlet_grid_node_(node_idx).count(obj_idx) > 0)
                            ++cell_dirichlet_node_num;
                    }
                    if(cell_dirichlet_node_num == cell_node_num)
                        continue;

                    Scalar closest_dist = (std::numeric_limits<Scalar>::max)();
                    unsigned int closest_obj_idx = 0;
                    //get closest kinematic object idx
                    for(unsigned int i = 0; i < (this->collidable_objects_).size(); ++i)
                    {
                        Scalar signed_distance = (this->collidable_objects_)[i]->signedDistance(corner_pos);
                        if(signed_distance < closest_dist)
                        {
                            closest_dist = signed_distance;
                            closest_obj_idx = i;
                        }
                    }
                    //get corner velocity
                    Vector<Scalar,Dim> corner_vel(0);
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                        corner_vel = domain_corner_velocity_[obj_idx][global_corner_idx];
                    else
                    {
                        for(unsigned int i = 0; i < this->corner_grid_pair_num_[obj_idx][particle_idx][corner_idx]; ++i)
                        {
                            const MPMInternal::NodeIndexWeightPair<Scalar,Dim> &pair = this->corner_grid_weight_[obj_idx][particle_idx][corner_idx][i];
                            Scalar weight = pair.weight_value;
                            Vector<Scalar,Dim> node_vel = this->gridVelocity(obj_idx,pair.node_idx);
                            corner_vel += weight*node_vel;
                        }
                    }
                    //corner in contact with collidable object
                    //compute the impulse on domain corner
                    //project the corner onto the surface of collidable object
                    Scalar corner_obj_dist = this->collidable_objects_[closest_obj_idx]->signedDistance(corner_pos);
                    Scalar collide_threshold = this->collidable_objects_[closest_obj_idx]->collideThreshold();
                    Vector<Scalar,Dim> impulse(0);
                    if(this->collidable_objects_[closest_obj_idx]->collide(corner_pos,corner_vel,impulse))
                    {
                        //compute the impulse
                        domain_corner_colliding_impulse_[obj_idx][global_corner_idx] = impulse;
                        //project the domain corner onto the surface of collidable object
                        Vector<Scalar,Dim> normal = this->collidable_objects_[closest_obj_idx]->normal(corner_pos);
                        corner_pos += (collide_threshold - corner_obj_dist) * normal;
                        //update corner position with new velocity
                        corner_pos += (corner_vel + impulse)*dt;
                        this->particle_domain_corners_[obj_idx][particle_idx][corner_idx] = corner_pos;
                        //mark the corner as colliding, its position won't be update by the grid velocity
                        is_colliding_domain_corner_[obj_idx][global_corner_idx] = 0x01;
                    }
                }
                //interpolate the impulse on domain corners to particles
                if(this->is_dirichlet_particle_[obj_idx][particle_idx])
                    continue; //skip dirichlet particle
                SolidParticle<Scalar,Dim> &particle = this->particle(obj_idx,particle_idx);
                Vector<Scalar,Dim> particle_vel = particle.velocity();
                Vector<Scalar,Dim> particle_impulse(0);
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx = particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    particle_impulse += particle_corner_weight_[obj_idx][particle_idx][corner_idx]*domain_corner_colliding_impulse_[obj_idx][global_corner_idx];
                }
                particle_vel += particle_impulse;
                particle.setVelocity(particle_vel);
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
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method)  //RobustCPDI2
        update_method->updateParticleInterpolationWeightWithEnrichment(weight_function,particle_domain_mesh_,is_enriched_domain_corner_,
                                                                       this->particle_grid_weight_and_gradient_,this->particle_grid_pair_num_,
                                                                       this->corner_grid_weight_,this->corner_grid_pair_num_);
    else
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method == NULL)
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
    //plugin operation
    MPMSolidPluginBase<Scalar,Dim> *plugin = NULL;
    for(unsigned int i = 0; i < this->plugins_.size(); ++i)
    {
        plugin = dynamic_cast<MPMSolidPluginBase<Scalar,Dim>*>(this->plugins_[i]);
        if(plugin)
            plugin->onUpdateParticleConstitutiveModelState(dt);
    }
    //update the particle deformation gradient
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            SquareMatrix<Scalar,Dim> particle_deform_grad;

            unsigned int enriched_corner_num = enrichedDomainCornerNum(obj_idx,particle_idx);
            if(enriched_corner_num > 0) //transient/enriched particle compute deformation gradient directly from domain shape (like in FEM)
                particle_deform_grad = update_method->computeParticleDeformationGradientFromDomainShape(obj_idx,particle_idx);
            else //ordinary particle update deformation gradient as: F^(n+1) = F^n + dt*(partial_vel_partial_X)
            {//Note: the gradient of velocity is with respect to reference configuration!!! In comparison with conventional MPM
                particle_deform_grad = particle->deformationGradient();
                SquareMatrix<Scalar,Dim> particle_vel_grad(0);
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    Vector<unsigned int,Dim> node_idx = (this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].node_idx);
                    Vector<Scalar,Dim> weight_gradient = (this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].gradient_value);
                    particle_vel_grad += this->gridVelocity(obj_idx,node_idx).outerProduct(weight_gradient);
                }
                particle_deform_grad += dt*particle_vel_grad;
            }

            particle->setDeformationGradient(particle_deform_grad);  //update deformation gradient (this deformation gradient might be inverted)
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
    //update particle velocity with a combination of FLIP/PIC from grid/corner velocity
    //some are interpolated from grid, some are from domain corner
    unsigned int corner_num = (Dim == 2) ? 4 : 8;
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            if(this->is_dirichlet_particle_[obj_idx][particle_idx])
                continue;//skip boundary particles
            //determine particle type
            unsigned int enriched_corner_num = enrichedDomainCornerNum(obj_idx,particle_idx);
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            Vector<Scalar,Dim> flip_vel = particle->velocity(), pic_vel(0);
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle get influence from grid
            {
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    Vector<unsigned int,Dim> node_idx = (this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].node_idx);
                    if(this->gridMass(obj_idx,node_idx) <= std::numeric_limits<Scalar>::epsilon())
                        continue;
                    Scalar weight = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i].weight_value;
                    Vector<Scalar,Dim> cur_grid_vel(0),grid_vel_before(0);
                    if(this->grid_velocity_(node_idx).find(obj_idx) != this->grid_velocity_(node_idx).end())
                        cur_grid_vel = this->grid_velocity_(node_idx)[obj_idx];
                    else
                        throw PhysikaException("Error in updateParticleVelocity!");
                    if(this->grid_velocity_before_(node_idx).find(obj_idx) != this->grid_velocity_before_(node_idx).end())
                        grid_vel_before = this->grid_velocity_before_(node_idx)[obj_idx];
                    else
                        throw PhysikaException("Error in updateParticleVelocity!");
                    flip_vel += weight*(cur_grid_vel-grid_vel_before);
                    pic_vel += weight*cur_grid_vel;
                }
            }
            if(enriched_corner_num > 0) //transient/enriched particle get influence from domain corner as well
            {
                for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
                {
                    unsigned int global_corner_idx =  particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
                    {
                        Scalar weight = particle_corner_weight_[obj_idx][particle_idx][corner_idx];
                        flip_vel += weight*(domain_corner_velocity_[obj_idx][global_corner_idx]-domain_corner_velocity_before_[obj_idx][global_corner_idx]);
                        pic_vel += weight*domain_corner_velocity_[obj_idx][global_corner_idx];
                    }
                }
            }
            Vector<Scalar,Dim> new_vel = (this->flip_fraction_) * flip_vel + (1 - this->flip_fraction_) * pic_vel;
            particle->setVelocity(new_vel);
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticlePosition(Scalar dt)
{
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method)  //RobustCPDI2
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
                    if(is_colliding_domain_corner_[obj_idx][global_corner_idx]) //colliding corner, its position has been determined in collision
                        continue;
                    if(is_enriched_domain_corner_[obj_idx][global_corner_idx]) //update with corner's velocity
                        new_corner_pos += domain_corner_velocity_[obj_idx][global_corner_idx]*dt;
                    else  //update with velocity from grid
                    {
                        for(unsigned int i = 0; i < this->corner_grid_pair_num_[obj_idx][particle_idx][corner_idx]; ++i)
                        {
                            const MPMInternal::NodeIndexWeightPair<Scalar,Dim> &pair = this->corner_grid_weight_[obj_idx][particle_idx][corner_idx][i];
                            Scalar weight = pair.weight_value;
                            Vector<Scalar,Dim> node_vel = this->gridVelocity(obj_idx,pair.node_idx);
                            new_corner_pos += weight*node_vel*dt;
                        }
                    }
                    this->particle_domain_corners_[obj_idx][particle_idx][corner_idx] = new_corner_pos;
                }
            }
        }
        //update particle position with RobustCPDI2
        update_method->updateParticlePosition(dt,this->is_dirichlet_particle_);
    }
    else
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::setPrincipalStretchThreshold(Scalar threshold)
{
    if(threshold < 0)
    {
        std::cerr<<"Warning: invalid principal threshold provided, default value (0.1) is used instead!\n";
        principal_stretch_threshold_ = 0.1;
    }
    else
        principal_stretch_threshold_ = threshold;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::enrichedParticles(unsigned int object_idx, std::vector<unsigned int> &enriched_particles) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    enriched_particles = enriched_particles_[object_idx];
}

template <typename Scalar, int Dim>
unsigned int InvertibleMPMSolid<Scalar,Dim>::enrichedDomainCornerNum(unsigned int object_idx, unsigned int particle_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if(particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = (Dim==2) ? 4 : 8;
    unsigned int enriched_corner_num = 0;
    for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
    {
        unsigned int global_corner_idx = particle_domain_mesh_[object_idx]->eleVertIndex(particle_idx,corner_idx);
        if(is_enriched_domain_corner_[object_idx][global_corner_idx])
            ++enriched_corner_num;
    }
    return enriched_corner_num;
}

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolid<Scalar, Dim>::domainCornerMass(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx) const
{
    if (object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    unsigned global_corner_idx = particle_domain_mesh_[object_idx]->eleVertIndex(particle_idx, corner_idx);
    return domain_corner_mass_[object_idx][global_corner_idx];
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> InvertibleMPMSolid<Scalar, Dim>::domainCornerVelocity(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx) const
{
    if (object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    unsigned global_corner_idx = particle_domain_mesh_[object_idx]->eleVertIndex(particle_idx, corner_idx);
    return domain_corner_velocity_[object_idx][global_corner_idx];
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar, Dim>::setDomainCornerVelocity(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx, const Vector<Scalar, Dim> &velocity)
{
    if (object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    unsigned global_corner_idx = particle_domain_mesh_[object_idx]->eleVertIndex(particle_idx, corner_idx);
    domain_corner_velocity_[object_idx][global_corner_idx] = velocity;
}

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolid<Scalar, Dim>::particleDomainCornerWeight(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx) const
{
    if (object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    return particle_corner_weight_[object_idx][particle_idx][corner_idx];
}

template <typename Scalar, int Dim>
Vector<Scalar, Dim> InvertibleMPMSolid<Scalar, Dim>::particleDomainCornerGradient(unsigned int object_idx, unsigned int particle_idx, unsigned int corner_idx) const
{
    if (object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if (particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (corner_idx >= corner_num)
        throw PhysikaException("corner index out of range!");
    return particle_corner_gradient_[object_idx][particle_idx][corner_idx];
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::enableEnrichment()
{
    enable_enrichment_ = true;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::disableEnrichment()
{
    enable_enrichment_ = false;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::enableEntireEnrichment()
{
    enable_entire_enrichment_ = true;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::disableEntireEnrichment()
{
    enable_entire_enrichment_ = false;
}

template <typename Scalar, int Dim>
Scalar InvertibleMPMSolid<Scalar,Dim>::enrichmentMetric(unsigned int object_idx, unsigned int particle_idx) const
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if(particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    return particle_enrich_metric_[object_idx][particle_idx];
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::setEnrichmentMetric(unsigned int object_idx, unsigned int particle_idx, Scalar metric)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    if(particle_idx >= this->particleNumOfObject(object_idx))
        throw PhysikaException("particle index out of range!");
    if(metric < 0)
    {
        std::cerr<<"Warning: negative metric provided, clamped to 0!\n";
        particle_enrich_metric_[object_idx][particle_idx] = 0;
    }
    else if(metric > 1)
    {
        std::cerr<<"Warning: metric clamped to 1!\n";
        particle_enrich_metric_[object_idx][particle_idx] = 1;
    }
    else
        particle_enrich_metric_[object_idx][particle_idx] = metric;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::setEnrichmentMetric(unsigned int object_idx, Scalar metric)
{
    if(object_idx >= this->objectNum())
        throw PhysikaException("object index out of range!");
    unsigned int particle_num = this->particleNumOfObject(object_idx);
    for(unsigned int particle_idx = 0; particle_idx < particle_num; ++particle_idx)
        setEnrichmentMetric(object_idx,particle_idx,metric);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridForwardEuler(Scalar dt)
{
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method == NULL)
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
    //explicit integration
    //integration on grid and domain corner
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            //determine particle type
            unsigned int enriched_corner_num = enrichedDomainCornerNum(obj_idx,particle_idx);
            if(enriched_corner_num == 0) //ordinary particle solve only on the grid
            {
                solveForParticleWithNoEnrichmentForwardEulerOnGrid(obj_idx,particle_idx,dt);
            }
            else //transient/enriched particle solve on domain corners
            {
                //solveForParticleWithEnrichmentForwardEulerViaQuadraturePoints(obj_idx,particle_idx,enriched_corner_num,dt); //compute the internal force on domain corner (and later map to grid) via quadrature points
                solveForParticleWithEnrichmentForwardEulerViaParticle(obj_idx,particle_idx,enriched_corner_num,dt); //compute the internal force on domain corner (and later map to grid) via particle
            }
        }
    }
    //apply gravity
    applyGravityOnEnrichedDomainCorner(dt);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridBackwardEuler(Scalar dt)
{
    if (this->objectNum() == 0) //no objects in scene
        return;
    //create linear system && solver
    if (invertible_mpm_system_ == NULL)
        invertible_mpm_system_ = new InvertibleMPMSolidLinearSystem<Scalar, Dim>(this);
    Vector<unsigned int, Dim> grid_node_num = this->grid_.nodeNum();
    if (invertible_system_rhs_ == NULL)
        invertible_system_rhs_ = new EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >(grid_node_num);
    if (invertible_system_x_ == NULL)
        invertible_system_x_ = new EnrichedMPMUniformGridGeneralizedVector<Vector<Scalar, Dim> >(grid_node_num);
    if (this->system_solver_ == NULL) //CG solver by default
        this->system_solver_ = new ConjugateGradientSolver<Scalar>();
    std::vector<Vector<unsigned int, Dim> > active_grid_nodes;
    std::vector<std::vector<unsigned int> > enriched_particles;
    std::vector<VolumetricMesh<Scalar, Dim>*> particle_domain_topology;
    solveOnGridForwardEuler(dt); //explicit solve used as rhs and initial guess
    unsigned int corner_num = Dim == 2 ? 4 : 8;
    if (this->contact_method_) //contact method is used, solve each object independently
    {
        enriched_particles.resize(1);
        particle_domain_topology.resize(1);
        for (unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        {
            this->activeGridNodes(obj_idx, active_grid_nodes);
            enrichedParticles(obj_idx, enriched_particles[0]);
            particle_domain_topology[0] = particle_domain_mesh_[obj_idx];
            invertible_system_rhs_->setActivePattern(active_grid_nodes, enriched_particles, particle_domain_topology);
            invertible_system_x_->setActivePattern(active_grid_nodes, enriched_particles, particle_domain_topology);
            invertible_mpm_system_->setActiveObject(obj_idx);
            for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
            {
                (*invertible_system_rhs_)[active_grid_nodes[i]] = this->gridVelocity(obj_idx, active_grid_nodes[i]);
                (*invertible_system_x_)[active_grid_nodes[i]] = (*invertible_system_rhs_)[active_grid_nodes[i]];
            }
            for (unsigned int i = 0; i < enriched_particles[0].size(); ++i)
            {
                for (unsigned int j = 0; j < corner_num; ++j)
                {
                    (*invertible_system_rhs_)(0, enriched_particles[0][i], j) = this->domainCornerVelocity(obj_idx, enriched_particles[0][i], j);
                    (*invertible_system_x_)(0, enriched_particles[0][i], j) = (*invertible_system_rhs_)(0, enriched_particles[0][i], j);
                }
            }
            //solve
            bool status = this->system_solver_->solve(*invertible_mpm_system_, *invertible_system_rhs_, *invertible_system_x_);
            if (status)
            {
                //apply the solve result only if implicit solve converges, otherwise explicit results are used
                for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
                    this->setGridVelocity(obj_idx, active_grid_nodes[i], (*invertible_system_x_)[active_grid_nodes[i]]);
                for (unsigned int i = 0; i < enriched_particles[0].size(); ++i)
                    for (unsigned int j = 0; j < corner_num; ++j)
                        this->setDomainCornerVelocity(obj_idx, enriched_particles[0][i], j, (*invertible_system_x_)(0, enriched_particles[0][i], j));
            }
        }
    }
    else //no contact method used, solve on single value grid
    {
        enriched_particles.resize(this->objectNum());
        particle_domain_topology.resize(this->objectNum());
        this->activeGridNodes(active_grid_nodes);
        for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
        {
            //all objects share the same velocity at one grid node
            (*invertible_system_rhs_)[active_grid_nodes[i]] = this->gridVelocity(0, active_grid_nodes[i]);
            (*invertible_system_x_)[active_grid_nodes[i]] = (*invertible_system_rhs_)[active_grid_nodes[i]];
        }
        for (unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        {
            enrichedParticles(obj_idx, enriched_particles[obj_idx]);
            particle_domain_topology[obj_idx] = particle_domain_mesh_[obj_idx];
            for (unsigned int i = 0; i < enriched_particles[obj_idx].size(); ++i)
            {
                for (unsigned int j = 0; j < corner_num; ++j)
                {
                    (*invertible_system_rhs_)(obj_idx, enriched_particles[obj_idx][i], j) = this->domainCornerVelocity(obj_idx, enriched_particles[obj_idx][i], j);
                    (*invertible_system_x_)(obj_idx, enriched_particles[obj_idx][i], j) = (*invertible_system_rhs_)(obj_idx, enriched_particles[obj_idx][i], j);
                }
            }
        }
        invertible_system_rhs_->setActivePattern(active_grid_nodes, enriched_particles, particle_domain_topology);
        invertible_system_x_->setActivePattern(active_grid_nodes, enriched_particles, particle_domain_topology);
        invertible_mpm_system_->setActiveObject(-1);
        //solve
        bool status = this->system_solver_->solve(*invertible_mpm_system_, *invertible_system_rhs_, *invertible_system_x_);
        if (status)
        {
            //apply the solve result only if implicit solve converges, otherwise explicit results are used
            for (unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
            {
                for (unsigned int i = 0; i < active_grid_nodes.size(); ++i)
                    this->setGridVelocity(obj_idx, active_grid_nodes[i], (*invertible_system_x_)[active_grid_nodes[i]]);
                for (unsigned int i = 0; i < enriched_particles[0].size(); ++i)
                    for (unsigned int j = 0; j < corner_num; ++j)
                        this->setDomainCornerVelocity(obj_idx, enriched_particles[obj_idx][i], j, (*invertible_system_x_)(obj_idx, enriched_particles[obj_idx][i], j));
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject()
{
    CPDIMPMSolid<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject();
    unsigned int corner_num = Dim==2 ? 4 : 8;
    unsigned int last_object_idx = this->objectNum() - 1;
    unsigned int particle_num_of_last_object = this->particleNumOfObject(last_object_idx);
    std::vector<Scalar> one_particle_corner_weight(corner_num,0);
    std::vector<std::vector<Scalar> > all_particle_corner_weight(particle_num_of_last_object,one_particle_corner_weight);
    particle_corner_weight_.push_back(all_particle_corner_weight);
    std::vector<Vector<Scalar,Dim>> one_particle_corner_gradient(corner_num);
    std::vector<std::vector<Vector<Scalar,Dim> > > all_particle_corner_gradient(particle_num_of_last_object,one_particle_corner_gradient);
    particle_corner_gradient_.push_back(all_particle_corner_gradient);
    std::vector<unsigned int> empty_enriched_particles;
    enriched_particles_.push_back(empty_enriched_particles);
    std::vector<typename DeformationDiagonalization<Scalar,Dim>::DiagonalizedDeformation> all_particle_diag_deform_grad(particle_num_of_last_object);
    particle_diagonalized_deform_grad_.push_back(all_particle_diag_deform_grad);
    std::vector<Scalar> enrich_metric(particle_num_of_last_object,default_enrich_metric_);
    particle_enrich_metric_.push_back(enrich_metric);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(unsigned int object_idx)
{
    CPDIMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(object_idx);
    unsigned int corner_num = Dim==2 ? 4 : 8;
    std::vector<Scalar> one_particle_corner_weight(corner_num,0);
    particle_corner_weight_[object_idx].push_back(one_particle_corner_weight);
    std::vector<Vector<Scalar,Dim> > one_particle_corner_gradient(corner_num);
    particle_corner_gradient_[object_idx].push_back(one_particle_corner_gradient);
    typename DeformationDiagonalization<Scalar,Dim>::DiagonalizedDeformation one_particle_diag_deform_grad;
    particle_diagonalized_deform_grad_[object_idx].push_back(one_particle_diag_deform_grad);
    particle_enrich_metric_[object_idx].push_back(default_enrich_metric_);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(unsigned int object_idx)
{
    CPDIMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(object_idx);
    typename std::vector<std::vector<std::vector<Scalar> > >::iterator iter1 = particle_corner_weight_.begin() + object_idx;
    particle_corner_weight_.erase(iter1);
    typename std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > >::iterator iter2 = particle_corner_gradient_.begin() + object_idx;
    particle_corner_gradient_.erase(iter2);
    std::vector<std::vector<unsigned int> >::iterator iter3 = enriched_particles_.begin() + object_idx;
    enriched_particles_.erase(iter3);
    typename std::vector<std::vector<typename DeformationDiagonalization<Scalar,Dim>::DiagonalizedDeformation> >::iterator iter4 = particle_diagonalized_deform_grad_.begin() + object_idx;
    particle_diagonalized_deform_grad_.erase(iter4);
    typename std::vector<std::vector<Scalar> >::iterator iter5 = particle_enrich_metric_.begin() + object_idx;
    particle_enrich_metric_.erase(iter5);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx)
{
    CPDIMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(object_idx,particle_idx);
    typename std::vector<std::vector<Scalar> >::iterator iter1 = particle_corner_weight_[object_idx].begin() + particle_idx;
    particle_corner_weight_[object_idx].erase(iter1);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter2 = particle_corner_gradient_[object_idx].begin() + particle_idx;
    particle_corner_gradient_[object_idx].erase(iter2);
    typename std::vector<typename DeformationDiagonalization<Scalar,Dim>::DiagonalizedDeformation>::iterator iter3 = particle_diagonalized_deform_grad_[object_idx].begin() + particle_idx;
    particle_diagonalized_deform_grad_[object_idx].erase(iter3);
    typename std::vector<Scalar>::iterator iter4 = particle_enrich_metric_[object_idx].begin() + particle_idx;
    particle_enrich_metric_[object_idx].erase(iter4);
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
            is_colliding_domain_corner_[obj_idx][domain_corner_idx] = 0x00;
            domain_corner_colliding_impulse_[obj_idx][domain_corner_idx] = zero_vel;
            domain_corner_mass_[obj_idx][domain_corner_idx] = 0;
            domain_corner_velocity_[obj_idx][domain_corner_idx] = zero_vel;
            domain_corner_velocity_before_[obj_idx][domain_corner_idx] = zero_vel;
        }
        enriched_particles_[obj_idx].clear(); //clear enriched particles
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
    is_colliding_domain_corner_.resize(obj_num);
    domain_corner_colliding_impulse_.resize(obj_num);
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
        is_colliding_domain_corner_[obj_idx].resize(vert_num);
        domain_corner_colliding_impulse_[obj_idx].resize(vert_num);
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
    PHYSIKA_ASSERT(obj_idx<this->objectNum());
    PHYSIKA_ASSERT(particle_idx<this->particleNumOfObject(obj_idx));
    //rule one: if the particle is in a cell with all dirichlet nodes, the particle cannot be enriched
    //the dirichlet boundary is correctly enforced in this way
    const Grid<Scalar,Dim> &grid = this->grid();
    Vector<unsigned int,Dim> grid_cell_num = grid.cellNum();
    SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
    Vector<unsigned int,Dim> cell_idx;
    Vector<Scalar,Dim> bias_in_cell;
    grid.cellIndexAndBiasInCell(particle->position(),cell_idx,bias_in_cell);
    Vector<unsigned int,Dim> node_idx;
    unsigned int cell_node_num = Dim == 2 ? 4 : 8;
    Vector<unsigned int,Dim> cell_node_dim(2);
    unsigned int cell_dirichlet_node_num = 0;
    for(unsigned int flat_idx = 0; flat_idx < cell_node_num; ++flat_idx)
    {
        node_idx = cell_idx + this->multiDimIndex(flat_idx,cell_node_dim);
        if(this->is_dirichlet_grid_node_(node_idx).count(obj_idx) > 0)
            ++cell_dirichlet_node_num;
    }
    if(cell_dirichlet_node_num == cell_node_num) //negotiable, not necessarily all nodes
        return false;
    //rule two: dirichlet particle is not enriched
    if(this->is_dirichlet_particle_[obj_idx][particle_idx])
        return false;

    //entire enrichment turned on, except those in rule dirichlet
    if(enable_entire_enrichment_)
        return true;
    //enrichment explicitly turned off
    if(!enable_enrichment_)
        return false;

    //rule three: enrich for ill-deformed particle
    //metric function: f = min(f1,f2)
    Scalar condition_value = 0; //the metric number for enrichment: 0~1, enrich~no-enrich
    Scalar condition_threshold = particle_enrich_metric_[obj_idx][particle_idx];
    const SquareMatrix<Scalar,Dim> &diag_deform_grad = particle_diagonalized_deform_grad_[obj_idx][particle_idx].diag_deform_grad;
    //f1 =min_stretch/max_stretch (inverse of condition number of F)
    Scalar f1 = 0;
    Scalar min_stretch = (std::numeric_limits<Scalar>::max)(), max_stretch = (std::numeric_limits<Scalar>::lowest)();
    for(unsigned int dim = 0; dim < Dim; ++dim)
    {
        if(diag_deform_grad(dim,dim) < principal_stretch_threshold_)  //already considered inverted, enrich
            return true;
        min_stretch = (min_stretch < diag_deform_grad(dim,dim)) ? min_stretch : diag_deform_grad(dim,dim);
        max_stretch = (max_stretch > diag_deform_grad(dim,dim)) ? max_stretch : diag_deform_grad(dim,dim);
    }
    f1 = min_stretch/max_stretch;
    //f2 = inverse condition number of the matrix that represents skew, decomposed from F
    //ref: Algebraic Mesh Quality Metrics
    SquareMatrix<Scalar,Dim> skew_deform = factorizeParticleSkewDeformation(obj_idx,particle_idx);
    SquareMatrix<Scalar,Dim> left_rotation,right_rotation;
    Vector<Scalar,Dim> singular_values;
    skew_deform.singularValueDecomposition(left_rotation,singular_values,right_rotation);
    min_stretch = (std::numeric_limits<Scalar>::max)();
    max_stretch = (std::numeric_limits<Scalar>::lowest)();
    for(unsigned int dim = 0; dim < Dim; ++dim)
    {
        min_stretch = (min_stretch < singular_values[dim]) ? min_stretch : singular_values[dim];
        max_stretch = (max_stretch > singular_values[dim]) ? max_stretch : singular_values[dim];
    }
    Scalar f2 = min_stretch/max_stretch;
    condition_value = min(f1,f2);
    if(condition_value < condition_threshold)
        return true;
    return false;
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleDomainEnrichState()
{
    //first precompute SVD of deformation gradient for each particle
    diagonalizeParticleDeformationGradient();
    //then check for each particle if the criteria is satisfied
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
                enriched_particles_[obj_idx].push_back(particle_idx); //store the enriched particle index
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
void InvertibleMPMSolid<Scalar,Dim>::computeParticleInterpolationWeightAndGradientInInitialDomain()
{
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method == NULL)
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            update_method->computeParticleInterpolationWeightInParticleDomain(obj_idx,particle_idx,particle_corner_weight_[obj_idx][particle_idx]);
            update_method->computeParticleInterpolationGradientInParticleDomain(obj_idx,particle_idx,particle_corner_gradient_[obj_idx][particle_idx]);
        }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveForParticleWithNoEnrichmentForwardEulerOnGrid(unsigned int obj_idx, unsigned int particle_idx, Scalar dt)
{
    PHYSIKA_ASSERT(obj_idx<this->objectNum());
    PHYSIKA_ASSERT(particle_idx<this->particleNumOfObject(obj_idx));
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method == NULL)
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
    SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
    Scalar particle_initial_volume = this->particle_initial_volume_[obj_idx][particle_idx];

    SquareMatrix<Scalar,Dim> deform_grad, diag_first_PiolaKirchoff_stress, first_PiolaKirchoff_stress;
    deform_grad = particle->deformationGradient();
    SquareMatrix<Scalar,Dim> left_rotation = particle_diagonalized_deform_grad_[obj_idx][particle_idx].left_rotation;
    SquareMatrix<Scalar,Dim> diag_deform_grad = particle_diagonalized_deform_grad_[obj_idx][particle_idx].diag_deform_grad;
    SquareMatrix<Scalar,Dim> right_rotation = particle_diagonalized_deform_grad_[obj_idx][particle_idx].right_rotation;
    //clamp the principal stretch to the threshold if it's compressed too severely
    for(unsigned int row = 0; row < Dim; ++row)
        if(diag_deform_grad(row,row) < principal_stretch_threshold_)
            diag_deform_grad(row,row) = principal_stretch_threshold_;
    //temporarily set the deformation gradient of the particle to the diagonalized one to compute the unrotated stress
    particle->setDeformationGradient(diag_deform_grad);
    // P = U*P^*V^T
    diag_first_PiolaKirchoff_stress = particle->firstPiolaKirchhoffStress();
    first_PiolaKirchoff_stress = left_rotation*diag_first_PiolaKirchoff_stress*(right_rotation.transpose());
    //recover the deformation gradient of the particle
    particle->setDeformationGradient(deform_grad);

    for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
    {
        const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i];
        if(this->is_dirichlet_grid_node_(pair.node_idx).count(obj_idx) > 0)
            continue; //skip grid nodes that are boundary condition
        Vector<Scalar,Dim> weight_gradient = pair.gradient_value; //gradient is to reference configuration
        if(this->grid_mass_(pair.node_idx)[obj_idx] <= std::numeric_limits<Scalar>::epsilon())
            continue; //skip grid nodes with near zero mass
        if(this->contact_method_)  //if contact method other than the inherent one is employed, update the grid velocity of each object independently
            this->grid_velocity_(pair.node_idx)[obj_idx] += dt*(-1)*particle_initial_volume*first_PiolaKirchoff_stress*weight_gradient/this->grid_mass_(pair.node_idx)[obj_idx];
        else  //otherwise, grid velocity of all objects that ocuppy the node get updated
        {
            if(this->is_dirichlet_grid_node_(pair.node_idx).size() > 0)
                continue;  //if for any involved object, this node is set as dirichlet, then the node is dirichlet for all objects
            for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = this->grid_velocity_(pair.node_idx).begin();
                vel_iter != this->grid_velocity_(pair.node_idx).end(); ++vel_iter)
                if(this->gridMass(vel_iter->first,pair.node_idx) > std::numeric_limits<Scalar>::epsilon())
                    vel_iter->second += dt*(-1)*particle_initial_volume*first_PiolaKirchoff_stress*weight_gradient/this->grid_mass_(pair.node_idx)[obj_idx];
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveForParticleWithEnrichmentForwardEulerViaQuadraturePoints(unsigned int obj_idx, unsigned int particle_idx,
                                                                                                   unsigned int enriched_corner_num, Scalar dt)
{
    PHYSIKA_ASSERT(obj_idx<this->objectNum());
    PHYSIKA_ASSERT(particle_idx<this->particleNumOfObject(obj_idx));
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method == NULL)
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
    //we assume the particle has enriched domain corners
    SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
    unsigned int corner_num = (Dim == 2) ? 4 : 8;
    //2x2(2D),2x2x2(3D) quadrature points per domain are used to evaluate the internal force on the domain corners
    //first get the quadrature points (different number for different dimension)
    std::vector<Vector<Scalar,Dim> > gauss_points;
    Scalar one_over_sqrt_3 = 1.0/sqrt(3.0);
    if(Dim == 2)
    {
        gauss_points.resize(4);
        for(unsigned int i = 0; i < 2; ++i)
            for(unsigned int j = 0; j < 2; ++j)
            {
                gauss_points[2*i+j][0] = (2.0*i-1)*one_over_sqrt_3;
                gauss_points[2*i+j][1] = (2.0*j-1)*one_over_sqrt_3;
            }
    }
    else if(Dim == 3)
    {
        gauss_points.resize(8);
        for(unsigned int i = 0; i < 2; ++i)
            for(unsigned int j = 0; j < 2; ++j)
                for(unsigned int k = 0; k < 2; ++k)
                {
                    gauss_points[4*i+2*j+k][0] = (2.0*i-1)*one_over_sqrt_3;
                    gauss_points[4*i+2*j+k][1] = (2.0*j-1)*one_over_sqrt_3;
                    gauss_points[4*i+2*j+k][2] = (2.0*k-1)*one_over_sqrt_3;
                }
    }
    else
        throw PhysikaException("Wrong dimension specified!");
    //now quadrature
    SquareMatrix<Scalar,Dim> deform_grad, left_rotation, diag_deform_grad, right_rotation,
        diag_first_PiolaKirchoff_stress, first_PiolaKirchoff_stress, particle_domain_jacobian_ref;
    Vector<Scalar,Dim> gauss_point, domain_shape_function_gradient_to_ref;
    Vector<unsigned int,Dim> corner_idx_nd, corner_dim(2);
    for(unsigned int gauss_idx = 0; gauss_idx < gauss_points.size(); ++gauss_idx)
    {
        gauss_point = gauss_points[gauss_idx];
        deform_grad = update_method->computeDeformationGradientAtPointInParticleDomain(obj_idx,particle_idx,gauss_point);
        deform_grad_diagonalizer_.diagonalizeDeformationGradient(deform_grad,left_rotation,diag_deform_grad,right_rotation);
        //clamp the principal stretch to the threshold if it's compressed too severely
        for(unsigned int row = 0; row < Dim; ++row)
            if(diag_deform_grad(row,row) < principal_stretch_threshold_)
                diag_deform_grad(row,row) = principal_stretch_threshold_;
        //temporarily set the deformation gradient of the particle to the diagonalized one to compute the unrotated stress
        particle->setDeformationGradient(diag_deform_grad);
        // P = U*P^*V^T
        diag_first_PiolaKirchoff_stress = particle->firstPiolaKirchhoffStress();
        first_PiolaKirchoff_stress = left_rotation*diag_first_PiolaKirchoff_stress*(right_rotation.transpose());
        //recover the deformation gradient of the particle
        particle->setDeformationGradient(deform_grad);
        //the jacobian matrix between the reference particle domain and primitive shape
        particle_domain_jacobian_ref = update_method->computeJacobianBetweenReferenceAndPrimitiveParticleDomain(obj_idx,particle_idx,gauss_point);
        Scalar jacobian_det = particle_domain_jacobian_ref.determinant();
        for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
        {
            corner_idx_nd = this->multiDimIndex(corner_idx,corner_dim);
            unsigned int global_corner_idx =  particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
            domain_shape_function_gradient_to_ref = update_method->computeShapeFunctionGradientToReferenceCoordinateAtPointInParticleDomain(obj_idx,particle_idx,corner_idx_nd,gauss_point);
            if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
            {
                if(domain_corner_mass_[obj_idx][global_corner_idx] <= std::numeric_limits<Scalar>::epsilon())
                    continue;
                domain_corner_velocity_[obj_idx][global_corner_idx] += dt*(-1)*first_PiolaKirchoff_stress*domain_shape_function_gradient_to_ref*jacobian_det;
            }
            else  //transient particles rasterize the internal force of non-enriched corners to the grid, solve on the grid
            {//domain of transient particles may be degenerated, hence the internal force must be first mapped from the gauss point to corner in the reference configuration,
                // then mapped from corner to grid in current configuration
                for(unsigned int i = 0; i < this->corner_grid_pair_num_[obj_idx][particle_idx][corner_idx]; ++i)
                {
                    const MPMInternal::NodeIndexWeightPair<Scalar,Dim> &pair = this->corner_grid_weight_[obj_idx][particle_idx][corner_idx][i];
                    if(this->is_dirichlet_grid_node_(pair.node_idx).count(obj_idx) > 0)
                        continue; //skip grid nodes that are boundary condition
                    Scalar corner_grid_weight = pair.weight_value;
                    if(this->grid_mass_(pair.node_idx)[obj_idx] <= std::numeric_limits<Scalar>::epsilon())
                        continue; //skip grid nodes with near zero mass
                    if(this->contact_method_)  //if contact method other than the inherent one is employed, update the grid velocity of each object independently
                        this->grid_velocity_(pair.node_idx)[obj_idx] += dt*(-1)*first_PiolaKirchoff_stress*corner_grid_weight*domain_shape_function_gradient_to_ref*jacobian_det/this->grid_mass_(pair.node_idx)[obj_idx];
                    else  //otherwise, grid velocity of all objects that ocuppy the node get updated
                    {
                        if(this->is_dirichlet_grid_node_(pair.node_idx).size() > 0)
                            continue;  //if for any involved object, this node is set as dirichlet, then the node is dirichlet for all objects
                        for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = this->grid_velocity_(pair.node_idx).begin();
                            vel_iter != this->grid_velocity_(pair.node_idx).end(); ++vel_iter)
                            if(this->gridMass(vel_iter->first,pair.node_idx) > std::numeric_limits<Scalar>::epsilon())
                                vel_iter->second += dt*(-1)*first_PiolaKirchoff_stress*corner_grid_weight*domain_shape_function_gradient_to_ref*jacobian_det/this->grid_mass_(pair.node_idx)[obj_idx];
                    }
                }
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::solveForParticleWithEnrichmentForwardEulerViaParticle(unsigned int obj_idx, unsigned int particle_idx,
                                                                                           unsigned int enriched_corner_num, Scalar dt)
{
    PHYSIKA_ASSERT(obj_idx<this->objectNum());
    PHYSIKA_ASSERT(particle_idx<this->particleNumOfObject(obj_idx));
    RobustCPDI2UpdateMethod<Scalar,Dim> *update_method = dynamic_cast<RobustCPDI2UpdateMethod<Scalar,Dim>*>(this->cpdi_update_method_);
    if(update_method == NULL)
        throw PhysikaException("Invertible MPM only supports RobustCPDI2!");
    //we assume the particle has enriched domain corners
    SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
    unsigned int corner_num = (Dim == 2) ? 4 : 8;
    //map internal force from particle to domain corner (then to grid node)
    SquareMatrix<Scalar,Dim> deform_grad, diag_first_PiolaKirchoff_stress, first_PiolaKirchoff_stress;
    Vector<unsigned int,Dim> corner_idx_nd, corner_dim(2);
    deform_grad = particle->deformationGradient();
    SquareMatrix<Scalar,Dim> left_rotation = particle_diagonalized_deform_grad_[obj_idx][particle_idx].left_rotation;
    SquareMatrix<Scalar,Dim> diag_deform_grad = particle_diagonalized_deform_grad_[obj_idx][particle_idx].diag_deform_grad;
    SquareMatrix<Scalar,Dim> right_rotation = particle_diagonalized_deform_grad_[obj_idx][particle_idx].right_rotation;
    //clamp the principal stretch to the threshold if it's compressed too severely
    for(unsigned int row = 0; row < Dim; ++row)
        if(diag_deform_grad(row,row) < principal_stretch_threshold_)
            diag_deform_grad(row,row) = principal_stretch_threshold_;
    //temporarily set the deformation gradient of the particle to the diagonalized one to compute the unrotated stress
    particle->setDeformationGradient(diag_deform_grad);
    // P = U*P^*V^T
    diag_first_PiolaKirchoff_stress = particle->firstPiolaKirchhoffStress();
    first_PiolaKirchoff_stress = left_rotation*diag_first_PiolaKirchoff_stress*(right_rotation.transpose());
    //recover the deformation gradient of the particle
    particle->setDeformationGradient(deform_grad);
    Scalar particle_initial_volume = this->particle_initial_volume_[obj_idx][particle_idx];
    for(unsigned int corner_idx = 0; corner_idx < corner_num; ++corner_idx)
    {
        corner_idx_nd = this->multiDimIndex(corner_idx,corner_dim);
        unsigned int global_corner_idx =  particle_domain_mesh_[obj_idx]->eleVertIndex(particle_idx,corner_idx);
        if(is_enriched_domain_corner_[obj_idx][global_corner_idx])
        {
            if(domain_corner_mass_[obj_idx][global_corner_idx] <= std::numeric_limits<Scalar>::epsilon())
                continue;
            domain_corner_velocity_[obj_idx][global_corner_idx] +=
                dt*(-1)*particle_initial_volume*first_PiolaKirchoff_stress*particle_corner_gradient_[obj_idx][particle_idx][corner_idx]/domain_corner_mass_[obj_idx][global_corner_idx];
        }
        else  //transient particles rasterize the internal force of non-enriched corners to the grid, solve on the grid
        {//domain of transient particles may be degenerated, hence the internal force must be first mapped from the particle to corner in the reference configuration,
            // then mapped from corner to grid in current configuration
            for(unsigned int i = 0; i < this->corner_grid_pair_num_[obj_idx][particle_idx][corner_idx]; ++i)
            {
                const MPMInternal::NodeIndexWeightPair<Scalar,Dim> &pair = this->corner_grid_weight_[obj_idx][particle_idx][corner_idx][i];
                if(this->is_dirichlet_grid_node_(pair.node_idx).count(obj_idx) > 0)
                    continue; //skip grid nodes that are boundary condition
                Scalar corner_grid_weight = pair.weight_value;
                if(this->grid_mass_(pair.node_idx)[obj_idx] <= std::numeric_limits<Scalar>::epsilon())
                    continue; //skip grid nodes with near zero mass
                if(this->contact_method_)  //if contact method other than the inherent one is employed, update the grid velocity of each object independently
                {
                    this->grid_velocity_(pair.node_idx)[obj_idx] +=
                        dt*(-1)*particle_initial_volume*first_PiolaKirchoff_stress*corner_grid_weight*particle_corner_gradient_[obj_idx][particle_idx][corner_idx]/this->grid_mass_(pair.node_idx)[obj_idx];
                }
                else  //otherwise, grid velocity of all objects that ocuppy the node get updated
                {
                    if(this->is_dirichlet_grid_node_(pair.node_idx).size() > 0)
                        continue;  //if for any involved object, this node is set as dirichlet, then the node is dirichlet for all objects
                    for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = this->grid_velocity_(pair.node_idx).begin();
                        vel_iter != this->grid_velocity_(pair.node_idx).end(); ++vel_iter)
                        if(this->gridMass(vel_iter->first,pair.node_idx) > std::numeric_limits<Scalar>::epsilon())
                        {
                            vel_iter->second +=
                                dt*(-1)*particle_initial_volume*first_PiolaKirchoff_stress*corner_grid_weight*particle_corner_gradient_[obj_idx][particle_idx][corner_idx]/this->grid_mass_(pair.node_idx)[obj_idx];
                        }
                }
            }
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::diagonalizeParticleDeformationGradient()
{
    SquareMatrix<Scalar,Dim> deform_grad;
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            typename DeformationDiagonalization<Scalar,Dim>::DiagonalizedDeformation &deform_grad_svd = particle_diagonalized_deform_grad_[obj_idx][particle_idx];
            deform_grad = particle->deformationGradient();
            deform_grad_diagonalizer_.diagonalizeDeformationGradient(deform_grad,deform_grad_svd.left_rotation,deform_grad_svd.diag_deform_grad,deform_grad_svd.right_rotation);
        }
}

template <typename Scalar, int Dim>
SquareMatrix<Scalar,Dim> InvertibleMPMSolid<Scalar,Dim>::factorizeParticleSkewDeformation(unsigned int obj_idx, unsigned int particle_idx) const
{
    PHYSIKA_ASSERT(obj_idx<this->objectNum());
    PHYSIKA_ASSERT(particle_idx<this->particleNumOfObject(obj_idx));
    const SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
    SquareMatrix<Scalar,Dim> F = particle->deformationGradient();
    SquareMatrix<Scalar,Dim> F_transpose_F = F.transpose()*F;
    SquareMatrix<Scalar,Dim> skew(0);
    if(Dim == 2)
    {
        Scalar denominator = sqrt(F_transpose_F(0,0)*F_transpose_F(1,1));
        skew(0,0) = 1;
        skew(0,1) = F_transpose_F(0,1)/denominator;
        skew(1,0) = 0;
        skew(1,1) = F.determinant()/denominator;
    }
    else if(Dim == 3)
    {
        Vector<Scalar,3> col_1, col_2, col_3;
        for(unsigned int i = 0; i < Dim; ++i)
        {
            col_1[i] = F(i,0);
            col_2[i] = F(i,1);
            col_3[i] = F(i,2);
        }
        Scalar denominator_0011 = sqrt(F_transpose_F(0,0)*F_transpose_F(1,1));
        Scalar denominator_0022 = sqrt(F_transpose_F(0,0)*F_transpose_F(2,2));
        Scalar denominator_22 = sqrt(F_transpose_F(2,2));
        Scalar col1_cross_col2 = (col_1.cross(col_2)).norm();
        skew(0,0) = 1;
        skew(1,0) = skew(2,0) = skew(2,1);
        skew(0,1) = F_transpose_F(0,1)/denominator_0011;
        skew(0,2) = F_transpose_F(0,2)/denominator_0022;
        skew(1,1) = col1_cross_col2/denominator_0011;
        skew(1,2) = (F_transpose_F(0,0)*F_transpose_F(1,2)-F_transpose_F(0,1)*F_transpose_F(0,2))/(denominator_0022*col1_cross_col2);
        skew(2,2) = F.determinant()/(denominator_22*col1_cross_col2);
    }
    else
        throw PhysikaException("Wrong dimension specified!");
    return skew;
}

//explicit instantiation
template class InvertibleMPMSolid<float,2>;
template class InvertibleMPMSolid<double,2>;
template class InvertibleMPMSolid<float,3>;
template class InvertibleMPMSolid<double,3>;

}  //end of namespace Physika
