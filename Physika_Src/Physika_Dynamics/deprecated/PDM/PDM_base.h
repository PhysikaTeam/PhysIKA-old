/*
 * @file PDM_base.h 
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

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_BASE_H

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Dynamics/Driver/driver_base.h"
#include "Physika_Dynamics/PDM/PDM_particle.h"


namespace Physika{

template<typename Scalar, int Dim> class VolumetricMesh;
template<typename Scalar, int Dim> class PDMStepMethodBase;

/*
 * Base class of PDM drivers.
 */

template<typename Scalar, int Dim>
class PDMBase: public DriverBase<Scalar>
{
public:
    PDMBase();
    PDMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~PDMBase();

    //virtual function 
    virtual void initConfiguration(const std::string & file_name);
    virtual void printConfigFileFormat();
    virtual void initSimulationData();
    virtual void advanceStep(Scalar dt);
    virtual Scalar computeTimeStep();
    virtual bool withRestartSupport() const;
    virtual void write(const std::string & file_name);
    virtual void read(const std::string & file_name);

    virtual void addPlugin(DriverPluginBase<Scalar>* plugin);

    //getters and setters
    Scalar gravity() const;
    Scalar delta(unsigned int par_idx) const;
    VolumetricMesh<Scalar, Dim> * mesh();
    unsigned int numSimParticles() const;

    Vector<Scalar, Dim> particleDisplacement(unsigned int par_idx) const;
    const Vector<Scalar, Dim> & particleRestPosition(unsigned int par_idx) const;
    const Vector<Scalar, Dim> & particleCurrentPosition(unsigned int par_idx) const;
    const Vector<Scalar, Dim> & particleVelocity(unsigned int par_idx) const;
    const Vector<Scalar, Dim> & particleForce(unsigned int par_idx) const;
    Scalar particleMass(unsigned int par_idx) const;
    PDMParticle<Scalar, Dim>  & particle(unsigned int par_idx);
    const PDMParticle<Scalar, Dim> & particle(unsigned int par_idx) const;

    const PDMStepMethodBase<Scalar, Dim> * stepMethod() const;
    PDMStepMethodBase<Scalar, Dim> * stepMethod();

    void setGravity(Scalar gravity);
    void setDelta(unsigned int par_idx, Scalar delta);
    void setHomogeneousDelta(Scalar delta);
    void setDeltaVec(const std::vector<Scalar> & delta_vec);
    void setMassViaHomogeneousDensity(Scalar density);
    
    void setAnisotropicMatrix(unsigned int par_idx, const SquareMatrix<Scalar, Dim> & anisotropic_matrix);
    void setHomogeneousAnisotropicMatrix(const SquareMatrix<Scalar, Dim> & anisotropic_matrix);

    void setHomogeneousEpStretchLimit(Scalar ep_stretch_limit);
    void setHomogeneousEbStretchLimit(Scalar eb_stretch_limit);

    void setParticleRestPos(unsigned int par_idx, const Vector<Scalar,Dim> & pos);

    void setParticleDisplacement(unsigned int par_idx, const Vector<Scalar,Dim> & u);
    void addParticleDisplacement(unsigned int par_idx, const Vector<Scalar,Dim> & u);
    void resetParticleDisplacement();

    void setParticleVelocity(unsigned int par_idx, const Vector<Scalar,Dim> & v);
    void addParticleVelocity(unsigned int par_idx, const Vector<Scalar,Dim> & v);
    void resetParticleVelocity();

    void setParticleForce(unsigned int par_idx, const Vector<Scalar, Dim> & f );
    void addParticleForce(unsigned int par_idx, const Vector<Scalar, Dim> & f);
    void resetParticleForce();

    void setParticleCurPos(unsigned int par_idx, const Vector<Scalar, Dim> & cur_pos);
    void addParticleCurPos(unsigned int par_idx, const Vector<Scalar, Dim> & u); // same as setParticleDisplacement
    void resetParticleCurPos(); // same as resetParticleDisplacement

    void setParticleMass(unsigned int par_idx, Scalar mass);
    void resetParticleMass();   // reset all particles mass to 1.0

    void setStepMethod(PDMStepMethodBase<Scalar, Dim> * step_method); //note: it will automatically bind driver & stepMethod
    
    void autoSetParticlesViaMesh(const std::string & file_name, Scalar max_delta_ratio, 
                                 const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix()); //max_delta = max_delta_ratio * average_edge_len

    void setParticlesViaMesh(const std::string & file_name, Scalar max_delta, bool use_hash_bin = false,
                             const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0,
                             const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix());

    void setParticlesViaMesh(VolumetricMesh<Scalar, Dim> * mesh, Scalar max_delta, bool use_hash_bin = false,
                             const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0,
                             const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix());

    void autoSetParticlesAtVertexViaMesh(const std::string & file_name, Scalar max_delta_ratio, 
                                         const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix()); //max_delta = max_delta_ratio * average_edge_len

    void setParticlesAtVertexViaMesh(const std::string & file_name, Scalar max_delta, bool use_hash_bin = false,
                                     const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0,
                                     const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix());

    void setParticlesAtVertexViaMesh(VolumetricMesh<Scalar, Dim> * mesh, Scalar max_delta, bool use_hash_bin = false,
                                     const Vector<Scalar, Dim> & start_bin_point = Vector<Scalar, Dim>(0), Scalar unify_spacing = 0.0, unsigned int unify_num = 0,
                                     const SquareMatrix<Scalar, Dim> & anisotropic_matrix = SquareMatrix<Scalar, Dim>::identityMatrix());

    void initParticleFamilyMember();
    void updateParticleFamilyMember();

    unsigned int timeStepId() const;

    void pauseSimulation();
    void forwardSimulation();
    bool isSimulationPause() const;

    void setWaitTimePerStep(unsigned int wait_time_per_step);

protected:
    void initParticleFamilyViaMaxDelta(Scalar max_delta, bool use_hash_bin, const Vector<Scalar, Dim> & start_bin_point, Scalar unify_spacing, unsigned int unify_num);
    void initVolumeWithMesh();
    void setParticleDirectNeighbor();

protected:
    PDMStepMethodBase<Scalar,Dim> * step_method_;             // step method to advance one step forward
    std::vector<PDMParticle<Scalar,Dim> > particles_;         // particle vector
    std::vector<Vector<Scalar,Dim> > particle_cur_pos_;       // current position of material points
    std::vector<Vector<Scalar,Dim> > particle_force_;         // force of material points

    Scalar gravity_;                                          // default: 9.8, gravity

    VolumetricMesh<Scalar, Dim> * mesh_; //volumetric mesh

    bool pause_simulation_;              //default: true
    unsigned int time_step_id_;          //initial: 0
    unsigned int wait_time_per_step_;    //default: 0, waiting time after each time step, in the unit of millisecond
};

}//end of namespace Physika;

#endif //PHYSIKA_DYNAMICS_PDM_PDM_BASE_H