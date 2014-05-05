/*
 * @file  tri_mesh.cpp
 * @brief Counterpart of Tetrahedral mesh in 2D.
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

#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Volumetric_Mesh/tri_mesh.h"

namespace Physika{

template <typename Scalar>
TriMesh<Scalar>::TriMesh()
{
}

template <typename Scalar>
TriMesh<Scalar>::TriMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements)
{
}

template <typename Scalar>
TriMesh<Scalar>::~TriMesh()
{
}

template <typename Scalar>
void TriMesh<Scalar>::printInfo() const
{
    std::cout<<"2D Triangle Mesh."<<std::endl;
}

template <typename Scalar>
int TriMesh<Scalar>::eleVolume(int ele_idx) const
{
}

template <typename Scalar>
bool TriMesh<Scalar>::containsVertex(int ele_idx, const Vector<Scalar,2> &pos) const
{
}

template <typename Scalar>
void TriMesh<Scalar>::interpolationWeights(int ele_idx, const Vector<Scalar,2> &pos, Scalar *weights) const
{
}

//explicit instantitation
template class TriMesh<float>;
template class TriMesh<double>;

}  //end of namespace Physika
