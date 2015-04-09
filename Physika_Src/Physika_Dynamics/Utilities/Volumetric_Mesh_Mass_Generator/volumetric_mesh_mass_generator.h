/*
 * @file volumetric_mesh_mass_generator.h 
 * @Brief given volumetric mesh and density, compute the mass needed for simulations that involve
 *            volumetric meshes, e.g., FEM.
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

#ifndef PHYSIKA_DYNAMICS_UTILITIES_VOLUMETRIC_MESH_MASS_GENERATOR_VOLUMETRIC_MESH_MASS_GENERATOR_H_
#define PHYSIKA_DYNAMICS_UTILITIES_VOLUMETRIC_MESH_MASS_GENERATOR_VOLUMETRIC_MESH_MASS_GENERATOR_H_

#include <vector>

namespace Physika{

template <typename Scalar, int Dim> class VolumetricMesh;
template <typename Scalar> class SparseMatrix;

template <typename Scalar, int Dim>
class VolumetricMeshMassGenerator
{
public:
    VolumetricMeshMassGenerator(){}
    ~VolumetricMeshMassGenerator(){}
    //static methods
    //lumped mass, uniform density
    static void generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, std::vector<Scalar> &lumped_mass);
    static void generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, SparseMatrix<Scalar> &lumped_mass);
    //lumped mass, element-wise density
    static void generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density, std::vector<Scalar> &lumped_mass);
    static void generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density, SparseMatrix<Scalar> &lumped_mass);
    //consistent mass matrix with uniform and element-wise density
    static void generateConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, SparseMatrix<Scalar> &consistent_mass);
    static void generateConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density, SparseMatrix<Scalar> &consistent_mass);
};

} //end of namespace Physika

#endif  //PHYSIKA_DYNAMICS_UTILITIES_VOLUMETRIC_MESH_MASS_GENERATOR_VOLUMETRIC_MESH_MASS_GENERATOR_H_
