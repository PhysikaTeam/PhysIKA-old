/*
 * @file fem_base.h 
 * @Brief Base class of FEM drivers, all FEM methods inherit from it.
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

#ifndef PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_
#define PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_

#include <string>
#include <vector>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
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
    FEMBase();
    FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file);
    FEMBase(unsigned int start_frame, unsigned int end_frame, Scalar frame_rate, Scalar max_dt, bool write_to_file,
        const VolumetricMesh<Scalar,Dim> &mesh);
    virtual ~FEMBase();

    //virtual methods for subclass to implement
    virtual void initConfiguration(const std::string &file_name)=0;
    virtual void printConfigFileFormat()=0;
    virtual void initSimulationData()=0;
    virtual void advanceStep(Scalar dt)=0;
    virtual Scalar computeTimeStep()=0;
    virtual bool withRestartSupport() const=0;
    virtual void write(const std::string &file_name)=0;
    virtual void read(const std::string &file_name)=0;
    virtual void addPlugin(DriverPluginBase<Scalar> *plugin)=0;
    
    //getters && setters
    Scalar gravity() const;
    void setGravity(Scalar gravity);
    void loadSimulationMesh(const std::string &file_name); //load the simulation mesh from file
    void setSimulationMesh(const VolumetricMesh<Scalar,Dim> &mesh);  //set the simulation mesh via an external mesh
    const VolumetricMesh<Scalar,Dim>* simulationMesh() const;  //return NULL if not set
    VolumetricMesh<Scalar,Dim>* simulationMesh();

    unsigned int numSimVertices() const; //number of simulation mesh vertices
    unsigned int numSimElements() const; //number of simulation mesh elements
    const Vector<Scalar,Dim>& vertexDisplacement(unsigned int vert_idx) const;
    void setVertexDisplacement(unsigned int vert_idx, const Vector<Scalar,Dim> &u);
    void resetVertexDisplacement(); //reset displacement of vertices to zero
    const Vector<Scalar,Dim>& vertexRestPosition(unsigned int vert_idx) const;
    Vector<Scalar,Dim> vertexCurrentPosition(unsigned int vert_idx) const;
    const Vector<Scalar,Dim>& vertexVelocity(unsigned int vert_idx) const;
    void setVertexVelocity(unsigned int vert_idx, const Vector<Scalar,Dim> &v);
    void resetVertexVelocity();
protected:
    void synchronizeDataWithSimulationMesh();  //synchronize related data when simulation mesh is changed (dimension of displacement vector, etc.)
    SquareMatrix<Scalar,Dim> deformationGradient(unsigned int ele_idx) const;  //compute the deformation gradient of given element, constant strain element
    void computeReferenceShapeMatrixInverse();  //compute the inverse of the reference shape matrix for each element, precomputation for deformation gradient
                                                //called only once when simulation mesh is set
protected:
    VolumetricMesh<Scalar,Dim> *simulation_mesh_;
    std::vector<Vector<Scalar,Dim> > vertex_displacements_;  //displacement of simulation mesh vertices
    std::vector<Vector<Scalar,Dim> > vertex_velocities_;  //velocities of simulation mesh vertices
    std::vector<SquareMatrix<Scalar,Dim> > reference_shape_matrix_inv_; //store precomputed data (inverse of Dm) for deformation gradient computation: F = Ds*inv(Dm)
    Scalar gravity_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_FEM_FEM_BASE_H_
