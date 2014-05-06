/*
 * @file  cubic_mesh.cpp
 * @brief hexahedral mesh class.
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
#include "Physika_Geometry/Volumetric_Mesh/cubic_mesh.h"

namespace Physika{

template <typename Scalar>
CubicMesh<Scalar>::CubicMesh()
{
}

template <typename Scalar>
CubicMesh<Scalar>::CubicMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements)
{
}

template <typename Scalar>
CubicMesh<Scalar>::~CubicMesh()
{
}

template <typename Scalar>
void CubicMesh<Scalar>::printInfo() const
{
    std::cout<<"Cubic Mesh."<<std::endl;
}

template <typename Scalar>
int CubicMesh<Scalar>::eleVolume(int ele_idx) const
{
    return 0;
}

template <typename Scalar>
bool CubicMesh<Scalar>::containsVertex(int ele_idx, const Vector<Scalar,3> &pos) const
{
    return false;
}

template <typename Scalar>
void CubicMesh<Scalar>::interpolationWeights(int ele_idx, const Vector<Scalar,3> &pos, Scalar *weights) const
{
}

//explicit instantitation
template class CubicMesh<float>;
template class CubicMesh<double>;

}  //end of namespace Physika














