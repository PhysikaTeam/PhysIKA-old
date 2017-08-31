/*
 * @file fem_base.h 
 * @Brief Base class of FEM drivers, all FEM methods inherit from it.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Dynamics/Driver/driver_base.h"

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;

/*
 * Base class of FEM drivers.
 * Two ways to set configurations before simulation:
 * 1. Various setters
 * 2. Load configuration from file
 */

template <typename Scalar, int Dim>
class FEMBase: public DriverBase<Scalar>
{
public:
    //different mass matrix types
    enum MassMatrixType{
        CONSISTENT_MASS,
        LUMPED_MASS
    };
public:
    FEMBase();
    FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    virtual ~FEMBase();

    //virtual methods for subclass to implement
    virtual void initConfiguration(const std::string &file_name)=0;
    virtual void printConfigFileFormat()=0;
    virtual void initSimulationData()=0;
    virtual void advanceStep(Scalar dt)=0;
    virtual bool withRestartSupport() const=0;
    virtual void write(const std::string &file_name)=0;
    virtual void read(const std::string &file_name)=0;
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin)=0;
    
    virtual Scalar computeTimeStep();
    
    //getters && setters
    unsigned int objectNum() const;
    void addObject(const VolumetricMesh<Scalar,Dim> &mesh, MassMatrixType mass_matrix_type);
    void removeObject(unsigned int object_idx);
    const VolumetricMesh<Scalar,Dim>& simulationMesh(unsigned int object_idx) const;
    VolumetricMesh<Scalar,Dim>& simulationMesh(unsigned int object_idx);
    Scalar cflConstant() const;
    void setCFLConstant(Scalar cfl); 
    Scalar soundSpeed() const;
    void setSoundSpeed(Scalar sound_speed);  
    Scalar gravity() const;
    void setGravity(Scalar gravity);

    unsigned int numSimVertices(unsigned int object_idx) const; //number of simulation mesh vertices
    unsigned int numSimElements(unsigned int object_idx) const; //number of simulation mesh elements
    Vector<Scalar,Dim> vertexDisplacement(unsigned int object_idx, unsigned int vert_idx) const;
    void setVertexDisplacement(unsigned int object_idx, unsigned int vert_idx, const Vector<Scalar,Dim> &u);
    void resetVertexDisplacement(unsigned int object_idx); //reset displacement of vertices to zero
    Vector<Scalar,Dim> vertexRestPosition(unsigned int object_idx, unsigned int vert_idx) const;
    Vector<Scalar,Dim> vertexCurrentPosition(unsigned int object_idx, unsigned int vert_idx) const;
    Vector<Scalar,Dim> vertexVelocity(unsigned int object_idx, unsigned int vert_idx) const;
    void setVertexVelocity(unsigned int object_idx, unsigned int vert_idx, const Vector<Scalar,Dim> &v);
    void resetVertexVelocity(unsigned int object_idx);
    Vector<Scalar,Dim> vertexExternalForce(unsigned int object_idx, unsigned int vert_idx) const;
    void setVertexExternalForce(unsigned int object_idx, unsigned int vert_idx, const Vector<Scalar,Dim> &f);
    void resetVertexExternalForce(unsigned int object_idx);
    
    //material density
    //set***Density() needs to be called to update density if volumetric mesh is updated
    unsigned int densityNum(unsigned int object_idx) const;
    void setHomogeneousDensity(unsigned int object_idx, Scalar density);
    void setRegionWiseDensity(unsigned int object_idx, const std::vector<Scalar> &density);
    void setElementWiseDensity(unsigned int object_idx, const std::vector<Scalar> &density);
    Scalar elementDensity(unsigned int object_idx, unsigned int ele_idx) const;
protected:
    virtual void applyGravity(unsigned int object_idx, Scalar dt);
    virtual void appendDataWithObject(); //append data associated with the newly added object
    virtual void removeDataWithObject(unsigned int object_idx);  //remove data associated with specific object
    void generateMassMatrix(unsigned int object_idx);
    Scalar maxVertexVelocityNorm(unsigned int object_idx) const;
    Scalar minElementCharacteristicLength(unsigned int object_idx) const;
    void clearSimulationMesh();
protected:
    //the object: simulation mesh and mass matrix type determine an FEM object
    std::vector<VolumetricMesh<Scalar,Dim>*> simulation_mesh_;
    std::vector<MassMatrixType> mass_matrix_type_;
    //data attached to the object
    std::vector<SparseMatrix<Scalar> > mass_matrix_;
    std::vector<std::vector<Vector<Scalar,Dim> > > vertex_displacements_;  
    std::vector<std::vector<Vector<Scalar,Dim> > > vertex_velocities_; 
    std::vector<std::vector<Vector<Scalar,Dim> > > vertex_external_forces_;
    std::vector<std::vector<Scalar> > material_density_;  //density: homogeneous, element-wise, or region-wise
    //time step computation with CFL condition
    Scalar cfl_num_;
    Scalar sound_speed_; //the sound speed in material
    Scalar gravity_; //magnitude of gravity, along negative y direction
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_
