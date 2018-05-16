/*
 * @file PDM_base.cpp 
 * @Basic PDMBase class. basic class of PDM
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include <iomanip>

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_IO/Volumetric_Mesh_IO/volumetric_mesh_io.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_grid_2d.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Collision_Methods/PDM_collision_method_grid_3d.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_base.h"

#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMBase<Scalar, Dim>::PDMBase()
    :DriverBase(),gravity_(9.8),pause_simulation_(true),mesh_(NULL),
    time_step_id_(0), wait_time_per_step_(0)
{

}

template <typename Scalar, int Dim>
PDMBase<Scalar, Dim>::PDMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file)
    :DriverBase<Scalar>(start_frame,end_frame,frame_rate,max_dt,write_to_file),gravity_(9.8),
    pause_simulation_(true),mesh_(NULL),time_step_id_(0), wait_time_per_step_(0)
{

}

template <typename Scalar, int Dim>
PDMBase<Scalar, Dim>::~PDMBase()
{
    delete this->mesh_;
}

// virtual function
template<typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::initConfiguration(const std::string & file_name)
{
    // to do
}

template<typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::printConfigFileFormat()
{
    // to do
}

template<typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::initSimulationData()
{
    // to do
}

template<typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::advanceStep(Scalar dt)
{
    //plugin operation, begin time step
    PDMPluginBase<Scalar, Dim> * plugin = NULL;
    for(unsigned int i=0; i<this->plugins_.size(); i++)
    {
        plugin = dynamic_cast<PDMPluginBase<Scalar, Dim>*>(this->plugins_[i]);
        if (plugin)
        {
            plugin->onBeginTimeStep(this->time_, dt);
        }
    }

    
    if(this->step_method_ == NULL)
    {
        std::cerr<<"No step method specified, program abort;\n";
        std::exit(EXIT_FAILURE);
    }

    
    if(this->pause_simulation_ == false)
    {
        Timer timer;
        timer.startTimer();

        std::system("cls");
        std::cout<<"**********************************************************\n";
        std::cout<<"TIME STEP : "<<this->time_step_id_<<std::endl;
        std::cout<<"dt :        "<<dt<<std::endl;
        std::cout<<"**********************************************************\n";

        //need further consideration
        if (dt > 1e-8)
        {
            this->step_method_->advanceStep(dt);
            this->time_step_id_++;
        }

        this->time_ += dt;

        timer.stopTimer();

        std::cout<<"**********************************************************\n";
        std::cout<<"TIME STEP COST: "<<timer.getElapsedTime()<<std::endl;
        std::cout<<"**********************************************************\n";

        Sleep(this->wait_time_per_step_);
    }
    
    //plugin operation, end time step
    for (unsigned int i=0; i<this->plugins_.size(); i++)
    {
        plugin = dynamic_cast<PDMPluginBase<Scalar, Dim>*>(this->plugins_[i]);
        if(plugin)
        {
            plugin->onEndTimeStep(this->time_, dt);
        }
    }
}

template<typename Scalar, int Dim>
Scalar PDMBase<Scalar, Dim>::computeTimeStep()
{
    // to do
    return this->max_dt_;
}

template<typename Scalar, int Dim>
bool PDMBase<Scalar, Dim>::withRestartSupport()const
{
    // to do
    return false;
}

template<typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::write(const std::string & file_name)
{
    // to do
}

template<typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::read(const std::string & file_name)
{
    // to do
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::addPlugin(DriverPluginBase<Scalar>* plugin)
{
    if(plugin==NULL)
    {
        std::cerr<<"Warning: NULL plugin provided, operation ignored!\n";
        return;
    }

    if(dynamic_cast<PDMPluginBase<Scalar, Dim> *>(plugin) == NULL)
    {
        std::cerr<<"Warning: Wrong type of plugin provided, operation ignored!\n";
        return;
    }

    plugin->setDriver(this);
    this->plugins_.push_back(plugin);
}

template <typename Scalar, int Dim>
Scalar PDMBase<Scalar, Dim>::gravity() const
{
    return this->gravity_;
}

template <typename Scalar, int Dim>
Scalar PDMBase<Scalar, Dim>::delta(unsigned int par_idx) const
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<"Particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    return this->particles_[par_idx].delta();

}

template <typename Scalar, int Dim>
VolumetricMesh<Scalar, Dim> * PDMBase<Scalar, Dim>::mesh()
{
    return this->mesh_;
}

template <typename Scalar, int Dim>
unsigned int PDMBase<Scalar, Dim>::numSimParticles() const
{
    return this->particles_.size();
}

template <typename Scalar, int Dim>
Vector<Scalar, Dim> PDMBase<Scalar, Dim>::particleDisplacement(unsigned int par_idx) const
{
    return this->particleCurrentPosition(par_idx) - this->particleRestPosition(par_idx);
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMBase<Scalar, Dim>::particleRestPosition(unsigned int par_idx) const
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particles_[par_idx].position();
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMBase<Scalar, Dim>::particleCurrentPosition(unsigned int par_idx) const
{
    if(par_idx >= this->particle_cur_pos_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particle_cur_pos_[par_idx];
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMBase<Scalar, Dim>::particleVelocity(unsigned int par_idx) const
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particles_[par_idx].velocity();
}

template <typename Scalar, int Dim>
const Vector<Scalar, Dim> & PDMBase<Scalar, Dim>::particleForce(unsigned int par_idx)const
{
    if(par_idx >= this->particle_force_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particle_force_[par_idx];
}

template <typename Scalar, int Dim>
Scalar PDMBase<Scalar, Dim>::particleMass(unsigned int par_idx) const
{
    if(par_idx >= this->particle_force_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particles_[par_idx].mass();
}

template <typename Scalar, int Dim>
PDMParticle<Scalar, Dim> & PDMBase<Scalar, Dim>::particle(unsigned int par_idx)
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particles_[par_idx];
}

template <typename Scalar, int Dim>
const PDMParticle<Scalar, Dim> & PDMBase<Scalar, Dim>::particle(unsigned int par_idx) const
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    return this->particles_[par_idx];
}

template <typename Scalar, int Dim>
const PDMStepMethodBase<Scalar, Dim> * PDMBase<Scalar, Dim>::stepMethod() const
{
    return this->step_method_;
}

template <typename Scalar, int Dim>
PDMStepMethodBase<Scalar, Dim> * PDMBase<Scalar, Dim>::stepMethod()
{
    return this->step_method_;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setGravity(Scalar gravity)
{
    this->gravity_ = gravity;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setDelta(unsigned int par_idx, Scalar delta)
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<"Particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->particles_[par_idx].setDelta(delta);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setHomogeneousDelta(Scalar delta)
{
    if (delta < 0)
    {
        std::cerr<<"error: delta should be greater than 0.\n";
        std::exit(EXIT_FAILURE);
    }
    for (unsigned int i=0; i<this->particles_.size(); i++)
    {
        this->particles_[i].setDelta(delta);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setDeltaVec(const std::vector<Scalar> & delta_vec)
{
    if(delta_vec.size() < this->particles_.size())
    {
        std::cerr<<"the size of delta_vec must be no less than the number of particles.\n ";
        std::exit(EXIT_FAILURE);
    }
    for (unsigned int i=0; i<this->particles_.size(); i++)
    {
        this->particles_[i].setDelta(delta_vec[i]);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setMassViaHomogeneousDensity(Scalar density)
{
    if (this->numSimParticles() == 0)
    {
        std::cerr<<"The driver contains no particle, please initialize from mesh!\n";
        std::exit(EXIT_FAILURE);
    }

    for (unsigned int par_id = 0; par_id < this->numSimParticles(); par_id++)
    {
        PDMParticle<Scalar, Dim> & particle = this->particle(par_id);
        particle.setMass(density*particle.volume());
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setAnisotropicMatrix(unsigned int par_idx, const SquareMatrix<Scalar, Dim> & anisotropic_matrix)
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<"Particle index out of range.\n";
        std::exit(EXIT_FAILURE);
    }
    this->particles_[par_idx].setAnistropicMatrix(anisotropic_matrix);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setHomogeneousAnisotropicMatrix(const SquareMatrix<Scalar, Dim> & anisotropic_matrix)
{
    for (unsigned int i=0; i<this->particles_.size(); i++)
        this->particles_[i].setAnistropicMatrix(anisotropic_matrix);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setHomogeneousEpStretchLimit(Scalar ep_stretch_limit)
{
    for (unsigned int par_id = 0; par_id < this->numSimParticles(); par_id++)
    {
        PDMParticle<Scalar, Dim> & particle = this->particle(par_id);

        std::list<PDMFamily<Scalar, Dim> > & family = particle.family();
        for (std::list<PDMFamily<Scalar, Dim> >::iterator iter = family.begin();  iter != family.end(); iter++)
            iter->setEpStretchLimit(ep_stretch_limit);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setHomogeneousEbStretchLimit(Scalar eb_stretch_limit)
{
    for (unsigned int par_id = 0; par_id < this->numSimParticles(); par_id++)
    {
        PDMParticle<Scalar, Dim> & particle = this->particle(par_id);

        std::list<PDMFamily<Scalar, Dim> > & family = particle.family();
        for (std::list<PDMFamily<Scalar, Dim> >::iterator iter = family.begin();  iter != family.end(); iter++)
            iter->setEbStretchLimit(eb_stretch_limit);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar,Dim>::setParticleRestPos(unsigned int par_idx, const Vector<Scalar,Dim> & pos)
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particles_[par_idx].setPosition(pos);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticleDisplacement(unsigned int par_idx, const Vector<Scalar, Dim> & u)
{
    if(par_idx >= this->particle_cur_pos_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particle_cur_pos_[par_idx] = this->particleRestPosition(par_idx)+u;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::addParticleDisplacement(unsigned int par_idx, const Vector<Scalar,Dim> & u)
{
    if(par_idx >= this->particle_cur_pos_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particle_cur_pos_[par_idx] += u;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::resetParticleDisplacement()
{
    for(unsigned int i=0; i<this->particle_cur_pos_.size(); i++)
    {
        this->particle_cur_pos_[i] = this->particleRestPosition(i);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticleVelocity(unsigned int par_idx, const Vector<Scalar,Dim> & v)
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particles_[par_idx].setVelocity(v);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::addParticleVelocity(unsigned int par_idx, const Vector<Scalar,Dim> & v)
{
    if(par_idx >= this->particles_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particles_[par_idx].addVelocity(v);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::resetParticleVelocity()
{
    for(unsigned int i=0; i<this->particles_.size(); i++)
    {
        this->particles_[i].setVelocity(Vector<Scalar,Dim>(0));
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticleForce(unsigned int par_idx, const Vector<Scalar, Dim> & f )
{
    if(par_idx >= this->particle_force_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particle_force_[par_idx] = f;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::addParticleForce(unsigned int par_idx, const Vector<Scalar, Dim> & f)
{
    if(par_idx >= this->particle_force_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particle_force_[par_idx] += f;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::resetParticleForce()
{
    for (unsigned int i=0; i<this->particle_force_.size(); i++)
    {
        this->particle_force_[i] = Vector<Scalar,Dim>(0);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticleCurPos(unsigned int par_idx, const Vector<Scalar, Dim> & cur_pos)
{
    if(par_idx >= this->particle_force_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particle_cur_pos_[par_idx] = cur_pos;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::addParticleCurPos(unsigned int par_idx, const Vector<Scalar, Dim> & u)
{
    this->addParticleDisplacement(par_idx,u);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::resetParticleCurPos()
{
    this->resetParticleDisplacement();
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticleMass(unsigned int par_idx, Scalar mass)
{
    if(par_idx >= this->particle_force_.size())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->particles_[par_idx].setMass(mass);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::resetParticleMass()
{
    for(unsigned int i=0; i<this->particles_.size(); i++)
    {
        this->particles_[i].setMass(1.0);
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::autoSetParticlesViaMesh(const std::string & file_name, Scalar max_delta_ratio,const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix())
{
    VolumetricMesh<Scalar, Dim> * mesh = VolumetricMeshIO<Scalar, Dim>::load(file_name);

    Vector<Scalar, Dim> bb_min_corner(std::numeric_limits<Scalar>::max());
    Vector<Scalar, Dim> bb_max_corner(std::numeric_limits<Scalar>::lowest());
    for (unsigned int vert_id = 0; vert_id < mesh->vertNum(); vert_id++)
    {
        Vector<Scalar, Dim> vert_pos = mesh->vertPos(vert_id);
        for (unsigned int i = 0; i < Dim; i++)
        {
            bb_min_corner[i] = min(bb_min_corner[i], vert_pos[i]);
            bb_max_corner[i] = max(bb_max_corner[i], vert_pos[i]);
        }
    }

    std::cout<<"bb_min_corner: "<<bb_min_corner<<std::endl;
    std::cout<<"bb_max_corner: "<<bb_max_corner<<std::endl;

    PHYSIKA_ASSERT(mesh->elementType() == VolumetricMeshInternal::TET);

    Scalar min_edge_len = std::numeric_limits<Scalar>::max();
    for (unsigned int ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, Dim> > ele_pos_vec;
        mesh->eleVertPos(ele_id, ele_pos_vec);

        for (unsigned int i = 0; i < 4; i++)
            for (unsigned int j = i+1; j < 4; j++)
            {
                Vector<Scalar, Dim> edge = ele_pos_vec[j] - ele_pos_vec[i];
                min_edge_len = min(min_edge_len, edge.norm());
            }
    }

    Scalar max_delta = max_delta_ratio * min_edge_len;

    std::cout<<"min_edge_len: "<<min_edge_len<<std::endl;
    std::cout<<"max_delta: "<<max_delta<<std::endl;

    Vector<Scalar, Dim> start_bin_point = bb_min_corner - 0.1*min_edge_len;
    Vector<Scalar, Dim> end_bin_point = bb_max_corner + 0.1*min_edge_len;

    Scalar max_length = std::numeric_limits<Scalar>::lowest();
    for (unsigned int i = 0; i < Dim; i++) max_length = max(max_length, end_bin_point[i] - start_bin_point[i]);

    Scalar unify_spacing = 1.1*max_delta;
    unsigned int unify_num = max_length/unify_spacing + 1;

    std::cout<<"start_bin_point: "<<start_bin_point<<std::endl;
    std::cout<<"end_bin_point: "<<end_bin_point<<std::endl;
    std::cout<<"unify_spacing: "<<unify_spacing<<std::endl;
    std::cout<<"unify_num: "<<unify_num<<std::endl;

    this->setParticlesViaMesh(mesh, max_delta, true, start_bin_point, unify_spacing, unify_num, anisotropic_matrix);
    
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticlesViaMesh(const std::string & file_name, Scalar max_delta, bool use_hash_bin = false, 
                                               const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0, 
                                               const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix())
{
    VolumetricMesh<Scalar, Dim> * mesh = VolumetricMeshIO<Scalar, Dim>::load(file_name);
    this->setParticlesViaMesh(mesh, max_delta, use_hash_bin, start_bin_point, unify_spacing, unify_num, anisotropic_matrix);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticlesViaMesh(VolumetricMesh<Scalar, Dim> * mesh, Scalar max_delta, bool use_hash_bin = false,
                                               const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0,
                                               const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix())
{
    this->mesh_ = mesh;

    this->particles_.clear();
    PDMParticle<Scalar, Dim> particle;
    for(unsigned int ele_id = 0; ele_id < this->mesh_->eleNum(); ele_id++)
    {
        Vector<Scalar, Dim> particle_pos(0.0);
        for (unsigned int ver_id = 0; ver_id < this->mesh_->eleVertNum(); ver_id++)
        {
            particle_pos += this->mesh_->eleVertPos(ele_id, ver_id);
        }
        particle_pos /= this->mesh_->eleVertNum();
        particle.setPosition(particle_pos);
        this->particles_.push_back(particle);
    }
    this->initVolumeWithMesh();
    this->setHomogeneousAnisotropicMatrix(anisotropic_matrix);

    std::cout<<"particle size: "<<this->particles_.size()<<std::endl;
    std::cout<<"particle vec is initialized!\n";

    this->particle_cur_pos_.clear();
    this->particle_force_.clear();
    this->particle_cur_pos_.resize(this->particles_.size());
    this->particle_force_.resize(this->particles_.size());

    //initialize current position of each particle
    this->resetParticleDisplacement();
    this->resetParticleForce();

    // initialize max family member via max delta
    this->initParticleFamilyViaMaxDelta(max_delta, use_hash_bin, start_bin_point, unify_spacing, unify_num);

    //defaultly set homogeneous delta and init particle family member
    this->setHomogeneousDelta(max_delta);
    this->initParticleFamilyMember();
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::autoSetParticlesAtVertexViaMesh(const std::string & file_name, Scalar max_delta_ratio,const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix())
{
    VolumetricMesh<Scalar, Dim> * mesh = VolumetricMeshIO<Scalar, Dim>::load(file_name);

    Vector<Scalar, Dim> bb_min_corner(std::numeric_limits<Scalar>::max());
    Vector<Scalar, Dim> bb_max_corner(std::numeric_limits<Scalar>::lowest());
    for (unsigned int vert_id = 0; vert_id < mesh->vertNum(); vert_id++)
    {
        Vector<Scalar, Dim> vert_pos = mesh->vertPos(vert_id);
        for (unsigned int i = 0; i < Dim; i++)
        {
            bb_min_corner[i] = min(bb_min_corner[i], vert_pos[i]);
            bb_max_corner[i] = max(bb_max_corner[i], vert_pos[i]);
        }
    }

    std::cout<<"bb_min_corner: "<<bb_min_corner<<std::endl;
    std::cout<<"bb_max_corner: "<<bb_max_corner<<std::endl;

    PHYSIKA_ASSERT(mesh->elementType() == VolumetricMeshInternal::TET);

    Scalar min_edge_len = std::numeric_limits<Scalar>::max();
    for (unsigned int ele_id = 0; ele_id < mesh->eleNum(); ele_id++)
    {
        std::vector<Vector<Scalar, Dim> > ele_pos_vec;
        mesh->eleVertPos(ele_id, ele_pos_vec);

        for (unsigned int i = 0; i < 4; i++)
            for (unsigned int j = i+1; j < 4; j++)
            {
                Vector<Scalar, Dim> edge = ele_pos_vec[j] - ele_pos_vec[i];
                min_edge_len = min(min_edge_len, edge.norm());
            }
    }

    Scalar max_delta = max_delta_ratio * min_edge_len;

    std::cout<<"min_edge_len: "<<min_edge_len<<std::endl;
    std::cout<<"max_delta: "<<max_delta<<std::endl;

    Vector<Scalar, Dim> start_bin_point = bb_min_corner - 0.1*min_edge_len;
    Vector<Scalar, Dim> end_bin_point = bb_max_corner + 0.1*min_edge_len;

    Scalar max_length = std::numeric_limits<Scalar>::lowest();
    for (unsigned int i = 0; i < Dim; i++) max_length = max(max_length, end_bin_point[i] - start_bin_point[i]);

    Scalar unify_spacing = 1.1*max_delta;
    unsigned int unify_num = max_length/unify_spacing + 1;

    std::cout<<"start_bin_point: "<<start_bin_point<<std::endl;
    std::cout<<"end_bin_point: "<<end_bin_point<<std::endl;
    std::cout<<"unify_spacing: "<<unify_spacing<<std::endl;
    std::cout<<"unify_num: "<<unify_num<<std::endl;

    this->setParticlesAtVertexViaMesh(mesh, max_delta, true, start_bin_point, unify_spacing, unify_num, anisotropic_matrix);

}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticlesAtVertexViaMesh(const std::string & file_name, Scalar max_delta, bool use_hash_bin = false, 
                                               const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0, 
                                               const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix())
{
    VolumetricMesh<Scalar, Dim> * mesh = VolumetricMeshIO<Scalar, Dim>::load(file_name);
    this->setParticlesAtVertexViaMesh(mesh, max_delta, use_hash_bin, start_bin_point, unify_spacing, unify_num, anisotropic_matrix);
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticlesAtVertexViaMesh(VolumetricMesh<Scalar, Dim> * mesh, Scalar max_delta, bool use_hash_bin = false,
                                                       const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0,
                                                       const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix())
{
    this->mesh_ = mesh;

    //init particles
    this->particles_.clear();
    PDMParticle<Scalar, Dim> particle;

    for (unsigned int ver_id = 0; ver_id < this->mesh_->vertNum(); ver_id++)
    {
        Vector<Scalar, Dim> particle_pos = this->mesh_->vertPos(ver_id);
        particle.setPosition(particle_pos);
        this->particles_.push_back(particle);
    }

    //reset volume
    for (unsigned int par_id = 0; par_id < this->particles_.size(); par_id++)
        this->particles_[par_id].setVolume(0.0);
    
    //init volume
    for (unsigned int ele_id = 0; ele_id < this->mesh_->eleNum(); ele_id++)
    {
        std::vector<unsigned int> ele_vertex_vec;
        this->mesh_->eleVertIndex(ele_id, ele_vertex_vec);
        Scalar ave_vol = this->mesh_->eleVolume(ele_id)/ele_vertex_vec.size();

        for (unsigned int ver_id = 0; ver_id < ele_vertex_vec.size(); ver_id++)
        {
            unsigned int par_id = ele_vertex_vec[ver_id];
            this->particles_[par_id].addVolume(ave_vol);
        }

    }

    this->setHomogeneousAnisotropicMatrix(anisotropic_matrix);

    std::cout<<"particle size: "<<this->particles_.size()<<std::endl;
    std::cout<<"particle vec is initialized!\n";

    this->particle_cur_pos_.clear();
    this->particle_force_.clear();
    this->particle_cur_pos_.resize(this->particles_.size());
    this->particle_force_.resize(this->particles_.size());

    //initialize current position of each particle
    this->resetParticleDisplacement();
    this->resetParticleForce();

    // initialize max family member via max delta
    this->initParticleFamilyViaMaxDelta(max_delta, use_hash_bin, start_bin_point, unify_spacing, unify_num);

    //defaultly set homogeneous delta and init particle family member
    this->setHomogeneousDelta(max_delta);
    this->initParticleFamilyMember();
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::initParticleFamilyMember()
{
    this->updateParticleFamilyMember();

    // initial size of family need to be set for each particle
    for (unsigned int i=0; i<this->particles_.size(); i++)
    {
        PDMParticle<Scalar, Dim> & particle = this->particles_[i];
        unsigned int init_num = particle.validFamilySize();
        particle.setInitFamilySize(init_num);    
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::updateParticleFamilyMember()
{
    for (unsigned int i=0; i<this->particles_.size(); i++)
    {
        this->particles_[i].updateFamilyMember();
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setStepMethod(PDMStepMethodBase<Scalar, Dim> * step_method)
{
    if(step_method == NULL)
    {
        std::cerr<<"NULL step method\n";
        std::exit(EXIT_FAILURE);
    }
    this->step_method_ = step_method;
    this->step_method_->setPDMDriver(this);
}

template <typename Scalar, int Dim>
unsigned int PDMBase<Scalar, Dim>::timeStepId() const
{
    return this->time_step_id_;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::pauseSimulation()
{
    this->pause_simulation_ = true;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::forwardSimulation()
{
    this->pause_simulation_ = false;
}

template <typename Scalar, int Dim>
bool PDMBase<Scalar, Dim>::isSimulationPause() const
{ 
    return this->pause_simulation_;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setWaitTimePerStep(unsigned int wait_time_per_step)
{
    this->wait_time_per_step_ = wait_time_per_step;
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::initParticleFamilyViaMaxDelta(Scalar max_delta, bool use_hash_bin, const Vector<Scalar, Dim> & start_bin_point, Scalar unify_spacing, unsigned int unify_num)
{
    // initialize the family of each material points 
    if (use_hash_bin)
    {
        PDMCollisionMethodGrid<Scalar,Dim> collision_method;
        collision_method.setUnifySpacing(unify_spacing);
        collision_method.setUnifyBinNum(unify_num);
        collision_method.setBinStartPoint(start_bin_point);
        collision_method.setDriver(this);
        collision_method.initParticleFamily(max_delta);   
    }
    else
    {
        for(unsigned int i =0; i<this->particles_.size(); i++)
        {
            Scalar delta_squared = max_delta*max_delta;
            for(unsigned int j= 0; j<this->particles_.size(); j++)
            {
                // if(|X_i - X_j| <= delta)
                if ( (i != j) &&(this->particleRestPosition(i)-this->particleRestPosition(j)).normSquared() <= delta_squared)
                {
                    Vector<Scalar, Dim> rest_relative_pos = this->particleRestPosition(j) - this->particleRestPosition(i);
                    Vector<Scalar, Dim> cur_relative_pos = this->particleCurrentPosition(j) - this->particleCurrentPosition(i);

                    const SquareMatrix<Scalar, Dim> anisotropic_matrix = this->particles_[i].anisotropicMatrix();
                    PDMFamily<Scalar, Dim> family(j, rest_relative_pos, anisotropic_matrix);
                    family.setCurRelativePos(cur_relative_pos);
                    this->particles_[i].addFamily(family);
                }
            }

            if (i%10000 == 0) std::cout<<"particle: "<<i<<std::endl;
        }
    }

    //set particle direct neighbor
    this->setParticleDirectNeighbor();
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::initVolumeWithMesh()
{
    PHYSIKA_ASSERT(this->particles_.size() == this->mesh_->eleNum());
    // set particle volume
    for (unsigned int i=0; i<this->mesh_->eleNum(); i++)
    {
        this->particles_[i].setVolume(this->mesh_->eleVolume(i));
    }
}

template <typename Scalar, int Dim>
void PDMBase<Scalar, Dim>::setParticleDirectNeighbor()
{
    std::cout<<"set particle direct neighbor......\n";

    #pragma omp parallel for
    for (long long par_id = 0; par_id < this->particles_.size(); par_id++)
    {
        std::list<PDMFamily<Scalar, Dim> > & family = this->particles_[par_id].family();
        std::list<PDMFamily<Scalar, Dim> >::iterator end_iter = family.end();

        for (std::list<PDMFamily<Scalar, Dim> >::iterator iter = family.begin(); iter != end_iter; iter++)
        {
            unsigned int fir_ele_id = par_id;
            unsigned int sec_ele_id = iter->id();

            std::vector<unsigned int> fir_ele_vert;
            std::vector<unsigned int> sec_ele_vert;
            this->mesh_->eleVertIndex(fir_ele_id, fir_ele_vert);
            this->mesh_->eleVertIndex(sec_ele_id, sec_ele_vert);
            PHYSIKA_ASSERT(fir_ele_vert.size() == sec_ele_vert.size());
            PHYSIKA_ASSERT(fir_ele_vert.size()==3 || fir_ele_vert.size()==4);

            unsigned int share_vertex_num = 0;
            for (unsigned int i=0; i<fir_ele_vert.size(); i++)
                for (unsigned int j=0; j<sec_ele_vert.size(); j++)
                {
                    if (fir_ele_vert[i] == sec_ele_vert[j])
                        share_vertex_num++;
                }
            PHYSIKA_ASSERT(share_vertex_num < fir_ele_vert.size());

            if (share_vertex_num == fir_ele_vert.size()-1)
                this->particles_[fir_ele_id].addDirectNeighbor(sec_ele_id);
        }

    }
}


//explicit instantiations
template class PDMBase<float,2>;
template class PDMBase<float,3>;
template class PDMBase<double,2>;
template class PDMBase<double,3>;

}// end of namespace Physika