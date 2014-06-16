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
#include "Physika_Core/Arrays/array.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Geometry/Volumetric_Meshes/cubic_mesh.h"

namespace Physika{

template <typename Scalar>
CubicMesh<Scalar>::CubicMesh():VolumetricMesh<Scalar,3>()
{
    (this->vert_per_ele_).clear();
    (this->vert_per_ele_).push_back(8);
}

template <typename Scalar>
CubicMesh<Scalar>::CubicMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements)
	:VolumetricMesh<Scalar, 3>(vert_num, vertices, ele_num, elements, 8)
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
VolumetricMeshInternal::ElementType CubicMesh<Scalar>::elementType() const
{
    return VolumetricMeshInternal::CUBIC;
}

template <typename Scalar>
int CubicMesh<Scalar>::eleVertNum() const
{
    return 8;
}

template <typename Scalar>
Scalar CubicMesh<Scalar>::eleVolume(unsigned int ele_idx) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"CubicMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Array< Vector<Scalar,3> > ele_vertices(8);
    for(int i = 0; i < 8; ++i)
        ele_vertices[i] = this->eleVertPos(ele_idx,i);
    //volume = |01 x 03 x 04|
    Vector<Scalar,3> fir_minus_0 = ele_vertices[1] - ele_vertices[0];
    Vector<Scalar,3> thi_minus_0 = ele_vertices[3] - ele_vertices[0];
    Vector<Scalar,3> fou_minus_0 = ele_vertices[4] - ele_vertices[0]; 
    return 1.0 * (fir_minus_0.norm() * thi_minus_0.norm() * fou_minus_0.norm() ) ;   
}

template <typename Scalar>
bool CubicMesh<Scalar>::containsVertex(unsigned int ele_idx, const Vector<Scalar,3> &pos) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"CubicMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
	Scalar weights[8];
    interpolationWeights(ele_idx,pos,weights);
    bool vert_in_ele = true;
	for(int i=0; i<8; ++i)
        if(weights[i] < 0)
            vert_in_ele = false;
    return vert_in_ele;
}

template <typename Scalar>
void CubicMesh<Scalar>::interpolationWeights(unsigned int ele_idx, const Vector<Scalar,3> &pos, Scalar *weights) const
{
/*we use trilinear interpolation 
 *Dx0 = (x-pos[0][0])/(pos[1][0]-pos[0][0]);
 *Dx1 = (pos[1][0]-x)/(pos[1][0]-pos[0][0]);
 *Dy0 = ......
 *Dy1 = ......
 *Dz0 =
 *Dz1 =
 */
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"CubicMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Array< Vector<Scalar,3> > ele_vertices(8);
    for(int i = 0; i < 8; ++i)
	ele_vertices[i] = this->eleVertPos(ele_idx,i);
    Scalar Dx0,Dx1,Dy0,Dy1,Dz0,Dz1;
    Dx0 = (pos[0] - ele_vertices[0][0])/(ele_vertices[1][0] - ele_vertices[0][0]);
    Dx1 = (ele_vertices[1][0] - pos[0])/(ele_vertices[1][0] - ele_vertices[0][0]);
    Dy0 = (pos[1] - ele_vertices[1][1])/(ele_vertices[2][1] - ele_vertices[1][1]);
    Dy1 = (ele_vertices[2][1] - pos[1])/(ele_vertices[2][1] - ele_vertices[1][1]);
    Dz0 = (pos[2] - ele_vertices[0][2])/(ele_vertices[4][2] - ele_vertices[0][2]);
    Dz1 = (ele_vertices[4][2] - pos[2])/(ele_vertices[4][2] - ele_vertices[0][2]);
    weights[0] = Dx1 * Dy1 * Dz1;
    weights[1] = Dx0 * Dy1 * Dz1;
    weights[2] = Dx0 * Dy0 * Dz1;
    weights[3] = Dx1 * Dy0 * Dz1;
    weights[4] = Dx1 * Dy1 * Dz0;
    weights[5] = Dx0 * Dy1 * Dz0;
    weights[6] = Dx0 * Dy0 * Dz0;
    weights[7] = Dx1 * Dy0 * Dz0;
}

//explicit instantitation
template class CubicMesh<float>;
template class CubicMesh<double>;

}  //end of namespace Physika
