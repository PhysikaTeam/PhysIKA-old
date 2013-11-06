/*
 * @file  tet_mesh.cpp
 * @brief Tetrahedral mesh class.
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
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Volumetric_Mesh/tet_mesh.h"

namespace Physika{

template <typename Scalar>
TetMesh<Scalar>::TetMesh()
{
}

template <typename Scalar>
TetMesh<Scalar>::TetMesh(int vert_num, const Scalar *vertices, int ele_num, const int *elements)
    :VolumetricMesh<Scalar,3>(vert_num,vertices,ele_num,elements,4)
{
}

template <typename Scalar>
TetMesh<Scalar>::~TetMesh()
{
}

template <typename Scalar>
void TetMesh<Scalar>::info() const
{
    std::cout<<"Tetrahedral Mesh."<<std::endl;
}

template <typename Scalar>
int TetMesh<Scalar>::eleVolume(int ele_idx) const
{
    return 0;
}

template <typename Scalar>
bool TetMesh<Scalar>::containsVertex(int ele_idx, const Vector<Scalar,3> &pos) const
{
    return false;
}

template <typename Scalar>
void TetMesh<Scalar>::interpolationWeights(int ele_idx, const Vector<Scalar,3> &pos, Scalar *weights) const
{
}

//explicit instantitation
template class TetMesh<float>;
template class TetMesh<double>;

}//end of namespace Physika
