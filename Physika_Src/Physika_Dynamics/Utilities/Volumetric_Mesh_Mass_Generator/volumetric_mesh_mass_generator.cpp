/*
 * @file volumetric_mesh_mass_generator.cpp 
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

#include <string>
#include <sstream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Matrices/sparse_matrix.h"
#include "Physika_Core/Matrices/sparse_matrix_iterator.h"
#include "Physika_Core/Matrices/matrix_MxN.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/volumetric_mesh_mass_generator.h"

namespace Physika{

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, std::vector<Scalar> &lumped_mass)
{
    SparseMatrix<Scalar> lumped_mass_mat;
    generateLumpedMass(volumetric_mesh,density,lumped_mass_mat);
    unsigned int vert_num = volumetric_mesh.vertNum();
    PHYSIKA_ASSERT(lumped_mass_mat.nonZeros() == vert_num);
    lumped_mass.resize(vert_num);
    SparseMatrixIterator<Scalar> mass_iter(lumped_mass_mat);
    while(mass_iter)
    {
        unsigned int row = mass_iter.row();
        unsigned int col = mass_iter.col();
        PHYSIKA_ASSERT(row == col);
        lumped_mass[row] = mass_iter.value();
        ++mass_iter;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, SparseMatrix<Scalar> &lumped_mass)
{
    SparseMatrix<Scalar> consistent_mass;
    generateConsistentMass(volumetric_mesh,density,consistent_mass);
    unsigned int vert_num = volumetric_mesh.vertNum();
    PHYSIKA_ASSERT(consistent_mass.rows() == vert_num);
    PHYSIKA_ASSERT(consistent_mass.cols() == vert_num);
    lumped_mass.resize(vert_num,vert_num);
    std::vector<Scalar> row_elements;
    for(unsigned int i =0; i < vert_num; ++i)
    {
        consistent_mass.rowElements(i,row_elements);
        Scalar sum = 0;
        for(unsigned int j = 0; j < row_elements.size(); ++j)
            sum += row_elements[j];
        lumped_mass.setEntry(i,i,sum);
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density,
                                                                 std::vector<Scalar> &lumped_mass,
                                                                 typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option)
{
    SparseMatrix<Scalar> lumped_mass_mat;
    generateLumpedMass(volumetric_mesh,density,lumped_mass_mat,density_option);
    unsigned int vert_num = volumetric_mesh.vertNum();
    PHYSIKA_ASSERT(lumped_mass_mat.nonZeros() == vert_num);
    lumped_mass.resize(vert_num);
    SparseMatrixIterator<Scalar> mass_iter(lumped_mass_mat);
    while(mass_iter)
    {
        unsigned int row = mass_iter.row();
        unsigned int col = mass_iter.col();
        PHYSIKA_ASSERT(row == col);
        lumped_mass[row] = mass_iter.value();
        ++mass_iter;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density,
                                                                 SparseMatrix<Scalar> &lumped_mass,
                                                                 typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option)
{
    SparseMatrix<Scalar> consistent_mass;
    generateConsistentMass(volumetric_mesh,density,consistent_mass,density_option);
    unsigned int vert_num = volumetric_mesh.vertNum();
    PHYSIKA_ASSERT(consistent_mass.rows() == vert_num);
    PHYSIKA_ASSERT(consistent_mass.cols() == vert_num);
    lumped_mass.resize(vert_num,vert_num);
    std::vector<Scalar> row_elements;
    for(unsigned int i =0; i < vert_num; ++i)
    {
        consistent_mass.rowElements(i,row_elements);
        Scalar sum = 0;
        for(unsigned int j = 0; j < row_elements.size(); ++j)
            sum += row_elements[j];
        lumped_mass.setEntry(i,i,sum);
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, SparseMatrix<Scalar> &consistent_mass)
{
    unsigned int ele_num = volumetric_mesh.eleNum();
    unsigned int vert_num = volumetric_mesh.vertNum();
    MatrixMxN<Scalar> ele_mass_mat;
    consistent_mass.resize(vert_num,vert_num);
    for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
    {
        generateElementConsistentMass(volumetric_mesh,ele_idx,density,ele_mass_mat);
        unsigned int ele_vert_num = volumetric_mesh.eleVertNum(ele_idx);
        for(unsigned int i =0; i < ele_vert_num; ++i)
        {
            unsigned int global_i = volumetric_mesh.eleVertIndex(ele_idx,i);
            for(unsigned int j = 0; j < ele_vert_num; ++j)
            {
                unsigned int global_j = volumetric_mesh.eleVertIndex(ele_idx,j);
                Scalar mass_buffer = consistent_mass(global_i,global_j);
                mass_buffer += ele_mass_mat(i,j);
                consistent_mass.setEntry(global_i,global_j,mass_buffer);
            }
        }
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density,
                                                                     SparseMatrix<Scalar> &consistent_mass,
                                                                     typename VolumetricMeshMassGenerator<Scalar,Dim>::DensityOption density_option)
{
    if(density_option != ELEMENT_WISE && density_option != REGION_WISE)
        throw PhysikaException("Unknown DesnityOption!");
    unsigned int ele_num = volumetric_mesh.eleNum();
    unsigned int vert_num = volumetric_mesh.vertNum();
    unsigned int region_num = volumetric_mesh.regionNum();
    if(density_option == ELEMENT_WISE && density.size() != ele_num)
        throw PhysikaException("Element-wise density vector doesn't match mesh element number!");
    if(density_option == REGION_WISE && density.size() != region_num)
        throw PhysikaException("Region-wise density vector doesn't match mesh region number!");
    MatrixMxN<Scalar> ele_mass_mat;
    consistent_mass.resize(vert_num,vert_num);
    for(unsigned int ele_idx = 0; ele_idx < ele_num; ++ele_idx)
    {
        Scalar ele_density;
        if(density_option == ELEMENT_WISE)
            ele_density = density[ele_idx];
        if(density_option == REGION_WISE)
        {
            int region_idx = volumetric_mesh.eleRegionIndex(ele_idx);
            if(region_idx < 0)
            {
                std::stringstream adaptor;
                adaptor<<ele_idx;
                std::string ele_idx_str;
                adaptor>>ele_idx_str;
                std::string err("Element ");
                err += ele_idx_str + std::string(" of volumetric mesh doesn't belong to any region, cannot determine it's density!");
                throw PhysikaException(err);
            }
            else
                ele_density = density[region_idx];
        }
        generateElementConsistentMass(volumetric_mesh,ele_idx,ele_density,ele_mass_mat);
        unsigned int ele_vert_num = volumetric_mesh.eleVertNum(ele_idx);
        for(unsigned int i =0; i < ele_vert_num; ++i)
        {
            unsigned int global_i = volumetric_mesh.eleVertIndex(ele_idx,i);
            for(unsigned int j = 0; j < ele_vert_num; ++j)
            {
                unsigned int global_j = volumetric_mesh.eleVertIndex(ele_idx,j);
                Scalar mass_buffer = consistent_mass(global_i,global_j);
                mass_buffer += ele_mass_mat(i,j);
                consistent_mass.setEntry(global_i,global_j,mass_buffer);
            }
        }
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, unsigned int ele_idx,
                                                                            Scalar density, MatrixMxN<Scalar> &ele_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(*tri_mesh,ele_idx,density,ele_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(*tet_mesh,ele_idx,density,ele_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(*quad_mesh,ele_idx,density,ele_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(*cubic_mesh,ele_idx,density,ele_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(const TriMesh<Scalar> &tri_mesh, unsigned int ele_idx,
                                                                            Scalar density, MatrixMxN<Scalar> &ele_mass)
{
    /*
     * consistent mass matrix of a triangle = mass/12*[M],
     * where [M] is a matrix:
     * M(i,j) = 2 if i == j
     * M(i,j) = 1 if i and j are end points of an edge
     *
     */
    if(ele_idx >= tri_mesh.eleNum())
        throw PhysikaException("volumetric mesh element index out of range!");
    Scalar factor = density*tri_mesh.eleVolume(ele_idx)/12.0;
    ele_mass.resize(3,3);
    for(unsigned int i = 0; i < 3; ++i)
        for(unsigned int j = 0; j < 3; ++j)
            ele_mass(i,j) = (i == j) ? 2*factor: factor;
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(const TetMesh<Scalar> &tet_mesh, unsigned int ele_idx,
                                                                            Scalar density, MatrixMxN<Scalar> &ele_mass)
{
    /*
     * consistent mass matrix of a tetrahedron = mass/20*[M],
     * where [M] is a matrix: [2,1,1,1;1,2,1,1;1,1,2,1;1,1,1,2] 
     */
    if(ele_idx >= tet_mesh.eleNum())
        throw PhysikaException("volumetric mesh element index out of range!");
    Scalar factor = density*tet_mesh.eleVolume(ele_idx)/20.0;
    ele_mass.resize(4,4);
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            ele_mass(i,j) = (i == j) ? 2*factor : factor;
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(const QuadMesh<Scalar> &quad_mesh, unsigned int ele_idx,
                                                                            Scalar density, MatrixMxN<Scalar> &ele_mass)
{
    /*
     * consistent mass matrix of a quad = mass/36*[M],
     * where [M] ia a matrix:
     * M(i,j) = 4, if i == j
     * M(i,j) = 2, if i and j are end points of an edge
     * M(i,j) = 1, if i and j are diagonal corners
     */
    if(ele_idx >= quad_mesh.eleNum())
        throw PhysikaException("volumetric mesh element index out of range!");
    ele_mass.resize(4,4);
    Scalar factor = density*quad_mesh.eleVolume(ele_idx)/36.0;
    Scalar stencil[16] = {4,2,1,2,
                          2,4,2,1,
                          1,2,4,2,
                          2,1,2,4};
    for(unsigned int i = 0; i < 4; ++i)
        for(unsigned int j = 0; j < 4; ++j)
            ele_mass(i,j) = factor * stencil[i*4+j];
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateElementConsistentMass(const CubicMesh<Scalar> &cubic_mesh, unsigned int ele_idx,
                                                                            Scalar density, MatrixMxN<Scalar> &ele_mass)
{
    /*
     * consistent mass matrix of a hexhahedron = mass/216*[M],
     * where [M] is a matrix:
     * M(i,j) = 8, if i == j
     * M(i,j) = 4, if i and j are end points of an edge
     * M(i,j) = 2, if i and j are two diagonal corners of a face
     * M(i,j) = 1, if i and j are the non-coplanar diagonal corners of the hexahedron
     *
     */
    if(ele_idx >= cubic_mesh.eleNum())
        throw PhysikaException("volumetric mesh element index out of range!");
    ele_mass.resize(8,8);
    Scalar factor = density*cubic_mesh.eleVolume(ele_idx)/216.0;
    Scalar stencil[64] = {8,4,2,4,4,2,1,2,
                          4,8,4,2,2,4,2,1,
                          2,4,8,4,1,2,4,2,
                          4,2,4,8,2,1,2,4,
                          4,2,1,2,8,4,2,4,
                          2,4,2,1,4,8,4,2,
                          1,2,4,2,2,4,8,4,
                          2,1,2,4,4,2,4,8};
    for(unsigned int i = 0; i < 8; ++i)
        for(unsigned int j = 0; j < 8; ++j)
            ele_mass(i,j) = factor * stencil[i*8+j];
}

//explicit instantiations
template class VolumetricMeshMassGenerator<float,2>;
template class VolumetricMeshMassGenerator<float,3>;
template class VolumetricMeshMassGenerator<double,2>;
template class VolumetricMeshMassGenerator<double,3>;

}  //end of namespace Physika
