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

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh_internal.h"
#include "Physika_Geometry/Volumetric_Meshes/tri_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/volumetric_mesh_mass_generator.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/tri_mesh_mass_generator.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/quad_mesh_mass_generator.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/tet_mesh_mass_generator.h"
#include "Physika_Dynamics/Utilities/Volumetric_Mesh_Mass_Generator/cubic_mesh_mass_generator.h"

namespace Physika{

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, std::vector<Scalar> &lumped_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        TriMeshMassGenerator<Scalar>::generateLumpedMass(*tri_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        TetMeshMassGenerator<Scalar>::generateLumpedMass(*tet_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        QuadMeshMassGenerator<Scalar>::generateLumpedMass(*quad_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        CubicMeshMassGenerator<Scalar>::generateLumpedMass(*cubic_mesh,density,lumped_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, SparseMatrix<Scalar> &lumped_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        TriMeshMassGenerator<Scalar>::generateLumpedMass(*tri_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        TetMeshMassGenerator<Scalar>::generateLumpedMass(*tet_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        QuadMeshMassGenerator<Scalar>::generateLumpedMass(*quad_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        CubicMeshMassGenerator<Scalar>::generateLumpedMass(*cubic_mesh,density,lumped_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density, std::vector<Scalar> &lumped_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        TriMeshMassGenerator<Scalar>::generateLumpedMass(*tri_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        TetMeshMassGenerator<Scalar>::generateLumpedMass(*tet_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        QuadMeshMassGenerator<Scalar>::generateLumpedMass(*quad_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        CubicMeshMassGenerator<Scalar>::generateLumpedMass(*cubic_mesh,density,lumped_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateLumpedMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density, SparseMatrix<Scalar> &lumped_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        TriMeshMassGenerator<Scalar>::generateLumpedMass(*tri_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        TetMeshMassGenerator<Scalar>::generateLumpedMass(*tet_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        QuadMeshMassGenerator<Scalar>::generateLumpedMass(*quad_mesh,density,lumped_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        CubicMeshMassGenerator<Scalar>::generateLumpedMass(*cubic_mesh,density,lumped_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, Scalar density, SparseMatrix<Scalar> &consistent_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        TriMeshMassGenerator<Scalar>::generateConsistentMass(*tri_mesh,density,consistent_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        TetMeshMassGenerator<Scalar>::generateConsistentMass(*tet_mesh,density,consistent_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        QuadMeshMassGenerator<Scalar>::generateConsistentMass(*quad_mesh,density,consistent_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        CubicMeshMassGenerator<Scalar>::generateConsistentMass(*cubic_mesh,density,consistent_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

template <typename Scalar, int Dim>
void VolumetricMeshMassGenerator<Scalar,Dim>::generateConsistentMass(const VolumetricMesh<Scalar,Dim> &volumetric_mesh, const std::vector<Scalar> &density, SparseMatrix<Scalar> &consistent_mass)
{
    VolumetricMeshInternal::ElementType ele_type = volumetric_mesh.elementType();
    switch(ele_type)
    {
    case VolumetricMeshInternal::TRI:
    {
        const TriMesh<Scalar> *tri_mesh = dynamic_cast<const TriMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tri_mesh);
        TriMeshMassGenerator<Scalar>::generateConsistentMass(*tri_mesh,density,consistent_mass);
        break;
    }
    case VolumetricMeshInternal::TET:
    {
        const TetMesh<Scalar> *tet_mesh = dynamic_cast<const TetMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(tet_mesh);
        TetMeshMassGenerator<Scalar>::generateConsistentMass(*tet_mesh,density,consistent_mass);
        break;
    }
    case VolumetricMeshInternal::QUAD:
    {
        const QuadMesh<Scalar> *quad_mesh = dynamic_cast<const QuadMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(quad_mesh);
        QuadMeshMassGenerator<Scalar>::generateConsistentMass(*quad_mesh,density,consistent_mass);
        break;
    }
    case VolumetricMeshInternal::CUBIC:
    {
        const CubicMesh<Scalar> *cubic_mesh = dynamic_cast<const CubicMesh<Scalar>*>(&volumetric_mesh);
        PHYSIKA_ASSERT(cubic_mesh);
        CubicMeshMassGenerator<Scalar>::generateConsistentMass(*cubic_mesh,density,consistent_mass);
        break;
    }
    default:
        throw PhysikaException("Unsupported element type for VolumetricMeshMassGenerator!");
        break;
    }
}

//explicit instantiations
template class VolumetricMeshMassGenerator<float,2>;
template class VolumetricMeshMassGenerator<float,3>;
template class VolumetricMeshMassGenerator<double,2>;
template class VolumetricMeshMassGenerator<double,3>;

}  //end of namespace Physika
