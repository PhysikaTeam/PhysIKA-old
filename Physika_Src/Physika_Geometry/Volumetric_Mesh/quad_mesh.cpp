/*
 * @file  quad_mesh.cpp
 * @brief Counterpart of hexahedral mesh in 2D.
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
#include "Physika_Geometry/Volumetric_Mesh/quad_mesh.h"

namespace Physika{

template <typename Scalar>
QuadMesh<Scalar>::QuadMesh()
{
}

template <typename Scalar>
QuadMesh<Scalar>::QuadMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements)
{
}

template <typename Scalar>
QuadMesh<Scalar>::~QuadMesh()
{
}

template <typename Scalar>
void QuadMesh<Scalar>::printInfo() const
{
    std::cout<<"2D Quad Mesh."<<std::endl;
}

template <typename Scalar>
int QuadMesh<Scalar>::eleVolume(int ele_idx) const
{
    return 0;
}

template <typename Scalar>
bool QuadMesh<Scalar>::containsVertex(int ele_idx, const Vector<Scalar,2> &pos) const
{
    return false;
}

template <typename Scalar>
void QuadMesh<Scalar>::interpolationWeights(int ele_idx, const Vector<Scalar,2> &pos, Scalar *weights) const
{
}

//explicit instantitation
template class QuadMesh<float>;
template class QuadMesh<double>;

}  //end of namespace Physika
