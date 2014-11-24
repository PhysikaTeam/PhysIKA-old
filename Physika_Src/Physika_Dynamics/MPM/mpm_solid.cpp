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
#include "Physika_Dynamics/MPM/MPM_Contact_Methods/mpm_solid_contact_method.h"
#include "Physika_Dynamics/MPM/mpm_solid.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid()
    :MPMSolidBase<Scalar,Dim>(), contact_method_(NULL)
{
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :MPMSolidBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file), contact_method_(NULL)
{
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::MPMSolid(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
                               const Grid<Scalar,Dim> &grid)
    :MPMSolidBase<Scalar,Dim>(start_frame,end_frame,frame_rate,max_dt,write_to_file),grid_(grid), contact_method_(NULL)
{
    synchronizeGridData();
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>::~MPMSolid()
{
    if(contact_method_)
        delete contact_method_;
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
Scalar MPMSolid<Scalar,Dim>::gridMass(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx) const
{
    if(object_idx < 0 || object_idx >= this->objectNum())
    {
        std::cerr<<"Error: object index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    bool valid_node_idx = isValidGridNodeIndex(node_idx);
    if(!valid_node_idx)
    {
        std::cerr<<"Error: invalid node index, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    typename std::map<unsigned int,Scalar>::const_iterator mass_map_iter = grid_mass_(node_idx).find(object_idx);
    if(mass_map_iter != grid_mass_(node_idx).end())
        return mass_map_iter->second;
    else
        return 0;
}

template <typename Scalar, int Dim>
Vector<Scalar,Dim> MPMSolid<Scalar,Dim>::gridVelocity(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx) const
{
    if(object_idx < 0 || object_idx >= this->objectNum())
    {
        std::cerr<<"Error: object index out of range, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    bool valid_node_idx = isValidGridNodeIndex(node_idx);
    if(!valid_node_idx)
    {
        std::cerr<<"Error: invalid node index, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    typename std::map<unsigned int,Vector<Scalar,Dim> >::const_iterator vel_map_iter = grid_velocity_(node_idx).find(object_idx);
    if(vel_map_iter != grid_velocity_(node_idx).end())
        return vel_map_iter->second;
    else
        return Vector<Scalar,Dim>(0);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::setGridVelocity(unsigned int object_idx, const Vector<unsigned int, Dim> &node_idx, const Vector<Scalar,Dim> &node_velocity)
{
    if(object_idx < 0 || object_idx >= this->objectNum())
    {
        std::cerr<<"Warning: object index out of range, operation ignored!\n";
        return;
    }
    bool valid_node_idx = isValidGridNodeIndex(node_idx);
    if(!valid_node_idx)
    {
        std::cerr<<"Warning: invalid node index, operation ignored!\n";
        return;
    }
    typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_map_iter = grid_velocity_(node_idx).find(object_idx);
    if(vel_map_iter == grid_velocity_(node_idx).end())
        grid_velocity_(node_idx).insert(std::make_pair(object_idx,node_velocity));
    else
        grid_velocity_(node_idx)[object_idx] = node_velocity;
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addDirichletGridNode(unsigned int object_idx, const Vector<unsigned int,Dim> &node_idx)
{
    if(object_idx < 0 || object_idx >= this->objectNum())
    {
        std::cerr<<"Warning: object index out of range, operation ignored!\n";
        return;
    }
    bool valid_node_idx = isValidGridNodeIndex(node_idx);
    if(!valid_node_idx)
    {
        std::cerr<<"Warning: invalid node index, operation ignored!\n";
        return;
    }
    is_dirichlet_grid_node_(node_idx).insert(object_idx);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::addDirichletGridNodes(unsigned int object_idx, const std::vector<Vector<unsigned int,Dim> > &node_idx)
{
    for(unsigned int i = 0; i < node_idx.size(); ++i)
        addDirichletGridNode(object_idx,node_idx[i]);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::setContactMethod(const MPMSolidContactMethod<Scalar,Dim> &contact_method)
{
    if(contact_method_)
        delete contact_method_;
    contact_method_ = contact_method.clone();
    contact_method_->setMPMDriver(this);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::resetContactMethod()
{
    if(contact_method_)
        delete contact_method_;
    contact_method_ = NULL;
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

    //rasterize mass and momentum of each object independently to grid
    resetGridData();
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
            {
                const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i];
                Scalar weight = pair.weight_value_;
                typename std::map<unsigned int,Scalar>::iterator iter = grid_mass_(pair.node_idx_).find(obj_idx);
                if(iter != grid_mass_(pair.node_idx_).end())
                    grid_mass_(pair.node_idx_)[obj_idx] += weight*particle->mass();
                else
                    grid_mass_(pair.node_idx_).insert(std::make_pair(obj_idx,weight*particle->mass()));
                if(is_dirichlet_grid_node_(pair.node_idx_).count(obj_idx)) //skip the velocity update of boundary nodes
                    continue;
                typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator iter2 = grid_velocity_(pair.node_idx_).find(obj_idx);
                if(iter2 != grid_velocity_(pair.node_idx_).end())
                    grid_velocity_(pair.node_idx_)[obj_idx] += weight*(particle->mass()*particle->velocity());
                else
                    grid_velocity_(pair.node_idx_).insert(std::make_pair(obj_idx,weight*(particle->mass()*particle->velocity())));
            }
        }
    }
    //determine active grid nodes according to the grid mass of each object
    Vector<unsigned int,Dim> grid_node_num = grid_.nodeNum();
    std::map<unsigned int,Vector<unsigned int,Dim> > active_node_idx_1d_nd_map;
    for(typename ArrayND<std::map<unsigned int,Scalar>,Dim>::Iterator iter = grid_mass_.begin(); iter != grid_mass_.end(); ++iter)
    {
        Vector<unsigned int,Dim> node_idx = iter.elementIndex();
        for(unsigned int i = 0; i < this->objectNum(); ++i)
        {
            if(gridMass(i,node_idx)>std::numeric_limits<Scalar>::epsilon())
            {
                unsigned int node_idx_1d = flatIndex(node_idx,grid_node_num);
                active_grid_node_.insert(std::make_pair(node_idx_1d,i));
                active_node_idx_1d_nd_map.insert(std::make_pair(node_idx_1d,node_idx));
            }
        }
    }
    //if no special contact algorithm is used, multi-value at a grid node must be converted to single value for all involved objects
    if(this->contact_method_==NULL)
    {
        for(typename std::map<unsigned int,Vector<unsigned int,Dim> >::iterator iter = active_node_idx_1d_nd_map.begin();
            iter != active_node_idx_1d_nd_map.end(); ++iter)
        {
            unsigned int node_idx_1d = iter->first;
            Vector<unsigned int,Dim> node_idx = iter->second;
            std::multimap<unsigned int,unsigned int>::iterator beg = active_grid_node_.lower_bound(node_idx_1d),
                end = active_grid_node_.upper_bound(node_idx_1d);
            std::multimap<unsigned int,unsigned int>::iterator cur = beg;
            Scalar mass_at_node = 0;
            Vector<Scalar,Dim> momentum_at_node(0);
            //accummulate values of all involved objects at this node
            while(cur != end)
            {
                mass_at_node += grid_mass_(node_idx)[cur->second];
                momentum_at_node += grid_velocity_(node_idx)[cur->second];
                ++cur;
            }
            //set all involved objects to uniform value at this node
            for(typename std::map<unsigned int,Scalar>::iterator mass_iter = grid_mass_(node_idx).begin(); mass_iter != grid_mass_(node_idx).end(); ++mass_iter)
                mass_iter->second = mass_at_node;
            //if for any involved object, this node is set as dirichlet, then the velocity of all objects is set as the prescribed velocity
            for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = grid_velocity_(node_idx).begin();
                vel_iter != grid_velocity_(node_idx).end(); ++vel_iter)
                if(is_dirichlet_grid_node_(node_idx).count(iter->first))
                {
                    momentum_at_node = vel_iter->second;
                    break;
                }
            //set the velocity
            for(typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator vel_iter = grid_velocity_(node_idx).begin();
                vel_iter != grid_velocity_(node_idx).end(); ++vel_iter)
                vel_iter->second = momentum_at_node;

        }
    }
    //compute grid's velocity, divide momentum by mass
    for(typename std::multimap<unsigned int,unsigned int>::iterator iter = active_grid_node_.begin(); iter != active_grid_node_.end(); ++iter)
    {
        unsigned int node_idx_1d = iter->first, object_idx = iter->second;
        Vector<unsigned int,Dim> node_idx = active_node_idx_1d_nd_map[node_idx_1d];
        if(is_dirichlet_grid_node_(node_idx).count(object_idx))
            continue; //skip grid nodes that are boundary condition
        grid_velocity_(node_idx)[object_idx] /= grid_mass_(node_idx)[object_idx];
        grid_velocity_before_(node_idx)[object_idx] = grid_velocity_(node_idx)[object_idx];  //buffer the grid velocity before any update
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

    //resolve contact on grid with specific contact method
    if(contact_method_)
    {
        std::vector<Vector<unsigned int,Dim> > potential_collide_nodes;
        std::vector<std::vector<unsigned int> > objects_at_node;
        std::set<unsigned int> involved_objects; //the set of objects involved in potential contact
        Vector<unsigned int,Dim> grid_node_num = grid_.nodeNum();
        std::multimap<unsigned int,unsigned int>::iterator iter = active_grid_node_.begin();
        while(iter != active_grid_node_.end())
        {
            unsigned int node_idx_1d = iter->first;
            unsigned int object_count = active_grid_node_.count(node_idx_1d);
            if(object_count > 1) //multiple objects at the node
            {
                Vector<unsigned int,Dim> node_idx = multiDimIndex(node_idx_1d,grid_node_num);
                std::vector<unsigned int> objects_at_this_node;
                while(object_count != 0) //element with equal key are stored in sequence in multimap
                {
                    if(is_dirichlet_grid_node_(node_idx).count(iter->second) == 0)  //not dirichlet node for the object
                    {
                        objects_at_this_node.push_back(iter->second);
                        involved_objects.insert(iter->second);
                    }
                    ++iter;
                    --object_count;
                }
                if(objects_at_this_node.size() > 0)
                {
                    potential_collide_nodes.push_back(node_idx);
                    objects_at_node.push_back(objects_at_this_node);
                }
            }
            else //no object or single object
                ++iter;
        }
        //compute the normal of the involved objects at the potential contact nodes
        std::vector<std::vector<Vector<Scalar,Dim> > > normal_at_node(potential_collide_nodes.size());
        for(unsigned int i = 0; i < normal_at_node.size(); ++i)
            normal_at_node[i].resize(objects_at_node[i].size());
        ArrayND<std::map<unsigned int,Vector<Scalar,Dim> >,Dim> grid_normal(grid_node_num);
        for(std::set<unsigned int>::iterator obj_iter = involved_objects.begin(); obj_iter != involved_objects.end(); ++obj_iter)
        {
            unsigned int obj_idx = *obj_iter;
            for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
            {
                SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
                for(unsigned int i = 0; i < this->particle_grid_pair_num_[obj_idx][particle_idx]; ++i)
                {
                    const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][particle_idx][i];
                    Vector<Scalar,Dim> weight_gradient = pair.gradient_value_;
                    typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator normal_iter = grid_normal(pair.node_idx_).find(obj_idx);
                    if(normal_iter != grid_normal(pair.node_idx_).end())
                        grid_normal(pair.node_idx_)[obj_idx] += weight_gradient*particle->mass();
                    else
                        grid_normal(pair.node_idx_).insert(std::make_pair(obj_idx,weight_gradient*particle->mass()));
                }
            }
        }
        for(unsigned int i = 0; i < potential_collide_nodes.size(); ++i)
        {
            Vector<unsigned int,Dim> node_idx = potential_collide_nodes[i];
            for(unsigned int j = 0; j < objects_at_node[i].size(); ++j)
            {
                unsigned int obj_idx = objects_at_node[i][j];
                //normalize normal
                typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator normal_iter = grid_normal(node_idx).find(obj_idx);
                if(normal_iter != grid_normal(node_idx).end())
                    normal_at_node[i][j] = grid_normal(node_idx)[obj_idx].normalize();
                else
                    PHYSIKA_ERROR("Error in computing grid normal!");
            }
        }
        //resolve contact
        contact_method_->resolveContact(potential_collide_nodes,objects_at_node,normal_at_node,dt);
    }
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
    
    //precompute the interpolation weights and gradients
    typedef UniformGridWeightFunctionInfluenceIterator<Scalar,Dim> InfluenceIterator;
    Vector<Scalar,Dim> grid_dx = (this->grid_).dX();
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int particle_idx = 0; particle_idx < this->particleNumOfObject(obj_idx); ++particle_idx)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][particle_idx];
            Vector<Scalar,Dim> particle_pos = particle->position();
            this->particle_grid_pair_num_[obj_idx][particle_idx] = 0;
            for(InfluenceIterator iter(this->grid_,particle_pos,*(this->weight_function_)); iter.valid(); ++iter)
            {
                Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
                Vector<Scalar,Dim> particle_to_node = particle_pos - (this->grid_).node(node_idx);
                for(unsigned int dim = 0; dim < Dim; ++dim)
                    particle_to_node[dim] /= grid_dx[dim];
                Vector<Scalar,Dim> weight_gradient = this->weight_function_->gradient(particle_to_node);
                Scalar weight = this->weight_function_->weight(particle_to_node);
                unsigned int i = this->particle_grid_pair_num_[obj_idx][particle_idx]++;
                (this->particle_grid_weight_and_gradient_)[obj_idx][particle_idx][i].node_idx_ = node_idx;
                (this->particle_grid_weight_and_gradient_)[obj_idx][particle_idx][i].weight_value_ = weight;
                (this->particle_grid_weight_and_gradient_)[obj_idx][particle_idx][i].gradient_value_ = weight_gradient;
            }
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

    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int i = 0; i < this->particleNumOfObject(obj_idx); ++i)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][i];
            SquareMatrix<Scalar,Dim> particle_vel_grad(0);
            for(unsigned int j = 0; j < this->particle_grid_pair_num_[obj_idx][i]; ++j)
            {
                Vector<unsigned int,Dim> node_idx = (this->particle_grid_weight_and_gradient_[obj_idx][i][j].node_idx_);
                Vector<Scalar,Dim> weight_gradient = (this->particle_grid_weight_and_gradient_[obj_idx][i][j].gradient_value_);
                particle_vel_grad += gridVelocity(obj_idx,node_idx).outerProduct(weight_gradient);
            }
            SquareMatrix<Scalar,Dim> particle_deform_grad = particle->deformationGradient();
            particle_deform_grad += dt*particle_vel_grad*particle_deform_grad;
            particle->setDeformationGradient(particle_deform_grad);  //update deformation gradient
            Scalar particle_vol = (particle_deform_grad.determinant())*(this->particle_initial_volume_[obj_idx][i]);
            particle->setVolume(particle_vol);  //update particle volume
        }
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
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {  
        for(unsigned int i = 0; i < this->particleNumOfObject(obj_idx); ++i)
        {
            if(this->is_dirichlet_particle_[obj_idx][i])
                continue;//skip boundary particles
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][i];
            Vector<Scalar,Dim> new_vel = particle->velocity();
            for(unsigned int j = 0; j < this->particle_grid_pair_num_[obj_idx][i]; ++j)
            {
                Vector<unsigned int,Dim> node_idx = this->particle_grid_weight_and_gradient_[obj_idx][i][j].node_idx_;
                Scalar weight = this->particle_grid_weight_and_gradient_[obj_idx][i][j].weight_value_;
                Vector<Scalar,Dim> cur_grid_vel(0),grid_vel_before(0);
                cur_grid_vel = gridVelocity(obj_idx,node_idx);
                if(grid_velocity_before_(node_idx).find(obj_idx)!=grid_velocity_before_(node_idx).end())
                    grid_vel_before = grid_velocity_before_(node_idx)[obj_idx];
                new_vel += weight*(cur_grid_vel-grid_vel_before);
            }
            particle->setVelocity(new_vel);
        }
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

    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int i = 0; i < this->particleNumOfObject(obj_idx); ++i)
        {
            if(this->is_dirichlet_particle_[obj_idx][i])
                continue;//skip boundary particles
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][i];
            Vector<Scalar,Dim> new_vel = particle->velocity();
            new_vel += this->particle_external_force_[obj_idx][i]/particle->mass()*dt;
            particle->setVelocity(new_vel);
        }
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
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int i = 0; i < this->particleNumOfObject(obj_idx); ++i)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][i];
            Vector<Scalar,Dim> new_pos = particle->position();
            if(this->is_dirichlet_particle_[obj_idx][i]) //for dirichlet particles, update position with prescribed velocity
                new_pos += this->particles_[obj_idx][i]->velocity()*dt;
            else
            {
                for(unsigned int j = 0; j < this->particle_grid_pair_num_[obj_idx][i]; ++j)
                {
                    Vector<unsigned int,Dim> node_idx = this->particle_grid_weight_and_gradient_[obj_idx][i][j].node_idx_;
                    Scalar weight = this->particle_grid_weight_and_gradient_[obj_idx][i][j].weight_value_;
                    new_pos += weight*gridVelocity(obj_idx,node_idx)*dt;
                }
            }
            particle->setPosition(new_pos);
        }
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
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::resetGridData()
{
    active_grid_node_.clear();
    for(typename Grid<Scalar,Dim>::NodeIterator iter = grid_.nodeBegin(); iter != grid_.nodeEnd(); ++iter)
    {
        Vector<unsigned int,Dim> node_idx = iter.nodeIndex();
        grid_mass_(node_idx).clear();
        for(unsigned int i = 0; i < this->objectNum(); ++i)
        {
            if(is_dirichlet_grid_node_(node_idx).count(i))
                continue; //skip grid nodes that are boundary condition
            //reset the velocity of object at the node
            typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator iter1 = grid_velocity_(node_idx).find(i);
            if(iter1 != grid_velocity_(node_idx).end())
                grid_velocity_(node_idx).erase(iter1);
            typename std::map<unsigned int,Vector<Scalar,Dim> >::iterator iter2 = grid_velocity_before_(node_idx).find(i);
            if(iter2 != grid_velocity_before_(node_idx).end())
                grid_velocity_before_(node_idx).erase(iter2);
        }
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
    Vector<unsigned int,Dim> grid_node_num = this->grid_.nodeNum();
    Vector<Scalar,Dim> gravity_vec(0);
    gravity_vec[1] = (-1)*(this->gravity_);
    for(std::multimap<unsigned int,unsigned int>::iterator iter = active_grid_node_.begin(); iter != active_grid_node_.end(); ++iter)
    {
        Vector<unsigned int,Dim> node_idx = multiDimIndex(iter->first,grid_node_num);
        for(unsigned int i = 0; i < this->objectNum(); ++i)
        {
            if(is_dirichlet_grid_node_(node_idx).count(i))
                continue; //skip grid nodes that are boundary condition
            if(gridMass(i,node_idx) > std::numeric_limits<Scalar>::epsilon()) //only update velocity of object that is active at this node
                grid_velocity_(node_idx)[i] += gravity_vec*dt;
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::synchronizeWithInfluenceRangeChange()
{
    //for each particle, preallocate space that can store weight/gradient of maximum
    //number of nodes in range
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    for(unsigned int i = 0; i < particle_grid_weight_and_gradient_.size(); ++i)
        for(unsigned int j = 0; j < particle_grid_weight_and_gradient_[i].size(); ++j)
            particle_grid_weight_and_gradient_[i][j].resize(max_num);
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
void MPMSolid<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject()
{
    MPMSolidBase<Scalar,Dim>::appendAllParticleRelatedDataOfLastObject();
    unsigned int last_object_idx = this->objectNum() - 1;
    unsigned int particle_num_of_last_object = this->particleNumOfObject(last_object_idx);
    //for each particle, preallocate space that can store weight/gradient of maximum
    //number of nodes in range
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > particle_max_num_weight_and_gradient_vec(max_num);
    std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > >
        object_max_num_weight_and_gradient_vec(particle_num_of_last_object,particle_max_num_weight_and_gradient_vec);
    particle_grid_weight_and_gradient_.push_back(object_max_num_weight_and_gradient_vec);
    std::vector<unsigned int> particle_grid_pair_num_vec(particle_num_of_last_object,0);
    particle_grid_pair_num_.push_back(particle_grid_pair_num_vec);
}
    
template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::appendLastParticleRelatedDataOfObject(unsigned int object_idx)
{
    MPMSolidBase<Scalar,Dim>::appendLastParticleRelatedDataOfObject(object_idx);
    unsigned int max_num = 1;
    for(unsigned int i = 0; i < Dim; ++i)
        max_num *= (this->weight_function_->supportRadius())*2+1;
    std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > particle_max_num_weight_and_gradient_vec(max_num);
    particle_grid_weight_and_gradient_[object_idx].push_back(particle_max_num_weight_and_gradient_vec);
    particle_grid_pair_num_[object_idx].push_back(0);
}
    
template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(unsigned int object_idx)
{
    MPMSolidBase<Scalar,Dim>::deleteAllParticleRelatedDataOfObject(object_idx);
    typename std::vector<std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > > >::iterator iter1 =
        particle_grid_weight_and_gradient_.begin() + object_idx;
    particle_grid_weight_and_gradient_.erase(iter1);
    std::vector<std::vector<unsigned int> >::iterator iter2 = particle_grid_pair_num_.begin() + object_idx;
    particle_grid_pair_num_.erase(iter2);
}
    
template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(unsigned int object_idx, unsigned int particle_idx)
{
    MPMSolidBase<Scalar,Dim>::deleteOneParticleRelatedDataOfObject(object_idx,particle_idx);
    typename std::vector<std::vector<MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> > >::iterator iter1 =
        particle_grid_weight_and_gradient_[object_idx].begin() + particle_idx;
    particle_grid_weight_and_gradient_[object_idx].erase(iter1);
    std::vector<unsigned int>::iterator iter2 = particle_grid_pair_num_[object_idx].begin() + particle_idx;
    particle_grid_pair_num_[object_idx].erase(iter2);
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::solveOnGridForwardEuler(Scalar dt)
{
    //explicit integration
    for(unsigned int obj_idx = 0; obj_idx < this->objectNum(); ++obj_idx)
    {
        for(unsigned int i = 0; i < this->particleNumOfObject(obj_idx); ++i)
        {
            SolidParticle<Scalar,Dim> *particle = this->particles_[obj_idx][i];
            for(unsigned int j = 0; j < this->particle_grid_pair_num_[obj_idx][i]; ++j)
            {
                const MPMInternal::NodeIndexWeightGradientPair<Scalar,Dim> &pair = this->particle_grid_weight_and_gradient_[obj_idx][i][j];
                if(is_dirichlet_grid_node_(pair.node_idx_).count(obj_idx))
                    continue; //skip grid nodes that are boundary condition
                Vector<Scalar,Dim> weight_gradient = pair.gradient_value_;
                SquareMatrix<Scalar,Dim> cauchy_stress = particle->cauchyStress();
                if(grid_mass_(pair.node_idx_)[obj_idx]>std::numeric_limits<Scalar>::epsilon())
                    grid_velocity_(pair.node_idx_)[obj_idx] += dt*(-1)*(particle->volume())*cauchy_stress*weight_gradient/grid_mass_(pair.node_idx_)[obj_idx];
            }
        }
    }
}

template <typename Scalar, int Dim>
void MPMSolid<Scalar,Dim>::solveOnGridBackwardEuler(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
unsigned int MPMSolid<Scalar,Dim>::flatIndex(const Vector<unsigned int,Dim> &index, const Vector<unsigned int,Dim> &dimension) const
{
    unsigned int flat_index = 0;
    Vector<unsigned int,Dim> vec = index;
    for(unsigned int i = 0; i < Dim; ++i)
    {
        for(unsigned int j = i+1; j < Dim; ++j)
            vec[i] *= dimension[j];
        flat_index += vec[i];
    }
    return flat_index;
}

template <typename Scalar, int Dim>
Vector<unsigned int,Dim> MPMSolid<Scalar,Dim>::multiDimIndex(unsigned int flat_index, const Vector<unsigned int,Dim> &dimension) const
{
    Vector<unsigned int,Dim> index(1);
    for(unsigned int i = 0; i < Dim; ++i)
    {
        for(unsigned int j = i+1; j < Dim; ++j)
            index[i] *= dimension[j];
        unsigned int temp = flat_index / index[i];
        flat_index = flat_index % index[i];
        index[i] = temp;
    }
    return index;
}

//explicit instantiations
template class MPMSolid<float,2>;
template class MPMSolid<float,3>;
template class MPMSolid<double,2>;
template class MPMSolid<double,3>;

}  //end of namespace Physika
