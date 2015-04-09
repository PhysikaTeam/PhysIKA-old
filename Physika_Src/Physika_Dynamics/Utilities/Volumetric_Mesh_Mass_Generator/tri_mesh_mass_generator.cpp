/*
 * @file tri_mesh_mass_generator.cpp 
 * @Brief given volumetric mesh with triangle element and density,
 *            compute the mass needed for simulations that involve
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

#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/tri_mesh_mass_generator.h"

namespace Physika{

template <typename Scalar>
void TriMeshMassGenerator<Scalar>::generateLumpedMass(const TriMesh<Scalar> &tri_mesh, Scalar density, std::vector<Scalar> &lumped_mass)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void TriMeshMassGenerator<Scalar>::generateLumpedMass(const TriMesh<Scalar> &tri_mesh, Scalar density, SparseMatrix<Scalar> &lumped_mass)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void TriMeshMassGenerator<Scalar>::generateLumpedMass(const TriMesh<Scalar> &tri_mesh, const std::vector<Scalar> &density, std::vector<Scalar> &lumped_mass)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void TriMeshMassGenerator<Scalar>::generateLumpedMass(const TriMesh<Scalar> &tri_mesh, const std::vector<Scalar> &density, SparseMatrix<Scalar> &lumped_mass)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void TriMeshMassGenerator<Scalar>::generateConsistentMass(const TriMesh<Scalar> &tri_mesh, Scalar density, SparseMatrix<Scalar> &consistent_mass)
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void TriMeshMassGenerator<Scalar>::generateConsistentMass(const TriMesh<Scalar> &tri_mesh, const std::vector<Scalar> &density, SparseMatrix<Scalar> &consistent_mass)
{
    throw PhysikaException("Not implemented!");
}

//explicit instantiations
template class TriMeshMassGenerator<float>;
template class TriMeshMassGenerator<double>;

}  //end of namespace Physika
