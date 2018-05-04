/*
 * @file  tri3d_mesh.cpp
 * @brief triangle mesh in 3d.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Arrays/array.h"
#include "Physika_Geometry/Volumetric_Meshes/tri3d_mesh.h"

namespace Physika{

template <typename Scalar>
Tri3DMesh<Scalar>::Tri3DMesh():VolumetricMesh<Scalar, 3>()
{
    (this->vert_per_ele_).clear();
    (this->vert_per_ele_).push_back(3);
}

template <typename Scalar>
Tri3DMesh<Scalar>::Tri3DMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements)
    :VolumetricMesh<Scalar,3>(vert_num, vertices, ele_num, elements, 3)
{
}

template <typename Scalar>
Tri3DMesh<Scalar>::Tri3DMesh(const Tri3DMesh<Scalar> &tri3d_mesh)
    :VolumetricMesh<Scalar,3>(tri3d_mesh)
{
}

template <typename Scalar>
Tri3DMesh<Scalar>::~Tri3DMesh()
{
}

template <typename Scalar>
Tri3DMesh<Scalar>& Tri3DMesh<Scalar>::operator= (const Tri3DMesh<Scalar> &tri3d_mesh)
{
    VolumetricMesh<Scalar,3>::operator= (tri3d_mesh);
    return *this;
}

template <typename Scalar>
Tri3DMesh<Scalar>* Tri3DMesh<Scalar>::clone() const
{
    return new Tri3DMesh<Scalar>(*this);
}

template <typename Scalar>
void Tri3DMesh<Scalar>::printInfo() const
{
    std::cout<<"3D Triangle Mesh."<<std::endl;
}

template <typename Scalar>
VolumetricMeshInternal::ElementType Tri3DMesh<Scalar>::elementType() const
{
    return VolumetricMeshInternal::TRI3D;
}

template <typename Scalar>
unsigned int Tri3DMesh<Scalar>::eleVertNum() const
{
    return 3;
}

template <typename Scalar>
Scalar Tri3DMesh<Scalar>::eleVolume(unsigned int ele_idx) const
{
    if(ele_idx>=this->ele_num_)
        throw PhysikaException("element index out of range!");
    
    Array< Vector<Scalar,3> > ele_vertices(3);
    for(unsigned int i = 0; i < 3; ++i)
        ele_vertices[i] = this->eleVertPos(ele_idx,i);
    Vector<Scalar,3> b_minus_a = ele_vertices[1] - ele_vertices[0];
    Vector<Scalar,3> c_minus_a = ele_vertices[2] - ele_vertices[0]; 
    return (b_minus_a.cross(c_minus_a)).norm()/2.0;
}

template <typename Scalar>
bool Tri3DMesh<Scalar>::containPoint(unsigned int ele_idx, const Vector<Scalar,3> &pos) const
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void Tri3DMesh<Scalar>::interpolationWeights(unsigned int ele_idx, const Vector<Scalar,3> &pos, std::vector<Scalar> &weights) const
{
    throw PhysikaException("Not implemented!");
}

template <typename Scalar>
void Tri3DMesh<Scalar>::generateBoundaryInformation()
{
    throw PhysikaException("Not implemented!");
}

//explicit instantiation
template class Tri3DMesh<float>;
template class Tri3DMesh<double>;

}//end of namespace Physika