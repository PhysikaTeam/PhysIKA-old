/*
 * @file invertible_mpm_solid.cpp 
 * @Brief a hybrid of FEM and CPDI2 for large deformation and invertible elasticity, uniform grid.
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

#include <limits>
#include "Physika_Core/Arrays/array_Nd.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/CPDI_Update_Methods/CPDI2_update_method.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_base.h"
#include "Physika_Dynamics/MPM/invertible_mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid()
    :CPDIMPMSolid<Scalar,Dim>()
{
    //only works with CPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<CPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :CPDIMPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file)
{
    //only works with CPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<CPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::InvertibleMPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                                                   const Grid<Scalar,Dim> &grid)
    :CPDIMPMSolid<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file,grid)
{
    //only works with CPDI2
    CPDIMPMSolid<Scalar,Dim>::template setCPDIUpdateMethod<CPDI2UpdateMethod<Scalar,Dim> >();
}

template <typename Scalar, int Dim>
InvertibleMPMSolid<Scalar,Dim>::~InvertibleMPMSolid()
{
    for(unsigned int i = 0; i < particle_domain_mesh_.size(); ++i)
        if(particle_domain_mesh_[i])
            delete particle_domain_mesh_[i];
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
    MPMSolid<Scalar,Dim>::initSimulationData();
    constructParticleDomainMesh();
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
            if(enriched_corner_num < corner_num) //ordinary particle && transient particle
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
                domain_corner_velocity_[obj_idx][corner_idx] /= domain_corner_mass_[obj_idx][corner_idx];
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
    CPDIMPMSolid<Scalar,Dim>::updateParticleInterpolationWeight();
    //TO DO: update the interpolation weight between particle and domain corner
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleConstitutiveModelState(Scalar dt)
{
    //TO DO:
}
    
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticleVelocity()
{
    //TO DO: some are interpolated from grid, some are from domain corner
}
 
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::updateParticlePosition(Scalar dt)
{
    //TO DO:
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
void InvertibleMPMSolid<Scalar,Dim>::solveOnGridForwardEuler(Scalar dt)
{
    //explicit integration
    
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
    particle_corner_gradient_.push_back(all_particle_corner_gradient);
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(unsigned int object_idx)
{
    CPDIMPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(object_idx);
    unsigned int corner_num = Dim==2 ? 4 : 8;
    std::vector<Scalar> one_particle_corner_weight(corner_num,0);
    std::vector<Vector<Scalar,Dim> > one_particle_corner_gradient(corner_num,Vector<Scalar,Dim>(0));
    particle_corner_weight_[object_idx].push_back(one_particle_corner_weight);
    particle_corner_gradient_[object_idx].push_back(one_particle_corner_gradient);
}
 
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(unsigned int object_idx)
{
    CPDIMPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(object_idx);
    typename std::vector<std::vector<std::vector<Scalar> > >::iterator iter1 = particle_corner_weight_.begin() + object_idx;
    particle_corner_weight_.erase(iter1);
    typename std::vector<std::vector<std::vector<Vector<Scalar,Dim> > > >::iterator iter2 = particle_corner_gradient_.begin() + object_idx;
    particle_corner_gradient_.erase(iter2);
}
 
template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx)
{
    CPDIMPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(object_idx,particle_idx);
    typename std::vector<std::vector<Scalar> >::iterator iter1 = particle_corner_weight_[object_idx].begin() + particle_idx;
    particle_corner_weight_[object_idx].erase(iter1);
    typename std::vector<std::vector<Vector<Scalar,Dim> > >::iterator iter2 = particle_corner_gradient_[object_idx].begin() + particle_idx;
    particle_corner_gradient_[object_idx].erase(iter2);
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
        }
    }
}

template <typename Scalar, int Dim>
void InvertibleMPMSolid<Scalar,Dim>::constructParticleDomainMesh()
{
    if(Dim == 2)
    {
    }
    else if(Dim == 3)
    {
    }
    else
        PHYSIKA_ERROR("Wrong dimension specified!");
}

template <typename Scalar, int Dim>
bool InvertibleMPMSolid<Scalar,Dim>::isEnrichCriteriaSatisfied(unsigned int obj_idx, unsigned int particle_idx) const
{
    return false;
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

//explicit instantiation
template class InvertibleMPMSolid<float,2>;
template class InvertibleMPMSolid<double,2>;
template class InvertibleMPMSolid<float,3>;
template class InvertibleMPMSolid<double,3>;

}  //end of namespace Physika
