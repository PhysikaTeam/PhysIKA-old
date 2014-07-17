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
#include "Physika_Core/Arrays/array.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Geometry/Volumetric_Meshes/quad_mesh.h"

namespace Physika{

template <typename Scalar>
QuadMesh<Scalar>::QuadMesh():VolumetricMesh<Scalar,2>()
{
    (this->vert_per_ele_).clear();
    (this->vert_per_ele_).push_back(4);
}

template <typename Scalar>
QuadMesh<Scalar>::QuadMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements)
	:VolumetricMesh<Scalar, 2>(vert_num, vertices, ele_num, elements, 4)
{
}

template <typename Scalar>
QuadMesh<Scalar>::QuadMesh(const QuadMesh<Scalar> &quad_mesh)
    :VolumetricMesh<Scalar,2>(quad_mesh)
{
}

template <typename Scalar>
QuadMesh<Scalar>::~QuadMesh()
{
}

template <typename Scalar>
QuadMesh<Scalar>& QuadMesh<Scalar>::operator= (const QuadMesh<Scalar> &quad_mesh)
{
    VolumetricMesh<Scalar,2>::operator= (quad_mesh);
    return *this;
}

template <typename Scalar>
void QuadMesh<Scalar>::printInfo() const
{
    std::cout<<"2D Quad Mesh."<<std::endl;
}

template <typename Scalar>
VolumetricMeshInternal::ElementType QuadMesh<Scalar>::elementType() const
{
    return VolumetricMeshInternal::QUAD;
}

template <typename Scalar>
unsigned int QuadMesh<Scalar>::eleVertNum() const
{
    return 4;
}

template <typename Scalar>
Scalar QuadMesh<Scalar>::eleVolume(unsigned int ele_idx) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"QuadMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Array< Vector<Scalar,2> > ele_vertices(4);
    for(unsigned int i = 0; i < 4; ++i)
	ele_vertices[i] = this->eleVertPos(ele_idx,i);
    //volume = 1/2*|ab x ac + ac x ad|
    Vector<Scalar,2> a_minus_d = ele_vertices[0] - ele_vertices[3];
    Vector<Scalar,2> b_minus_d = ele_vertices[1] - ele_vertices[3];
    Vector<Scalar,2> c_minus_d = ele_vertices[2] - ele_vertices[3]; 
    return 1.0/2*abs((b_minus_d.cross(a_minus_d)) + (b_minus_d.cross(c_minus_d)));
}

template <typename Scalar>
bool QuadMesh<Scalar>::containsVertex(unsigned int ele_idx, const Vector<Scalar,2> &pos) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"QuadMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Scalar weights[4];
    interpolationWeights(ele_idx,pos,weights);
    bool vert_in_ele = (weights[0]>=0) && (weights[1]>=0) && (weights[2]>=0 && (weights[3]>=0));
    return vert_in_ele;    
}

template <typename Scalar>
void QuadMesh<Scalar>::interpolationWeights(unsigned int ele_idx, const Vector<Scalar,2> &pos, Scalar *weights) const
{
/*we use bilinear interpolation 
 *Dx0 = (x-pos[0][0])/(pos[1][0]-pos[0][0]);
 *Dx1 = (pos[1][0]-x)/(pos[1][0]-pos[0][0]);
 *Dy0 = ......
 *Dy1 = ......
 *
 */
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"QuadMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Array< Vector<Scalar,2> > ele_vertices(4);
    for(unsigned int i = 0; i < 4; ++i)
        ele_vertices[i] = this->eleVertPos(ele_idx,i);
    Scalar Dx0,Dx1,Dy0,Dy1;
    Dx0 = (pos[0] - ele_vertices[0][0])/(ele_vertices[1][0] - ele_vertices[0][0]);
    Dx1 = (ele_vertices[1][0] - pos[0])/(ele_vertices[1][0] - ele_vertices[0][0]);
    Dy0 = (pos[1] - ele_vertices[1][1])/(ele_vertices[2][1] - ele_vertices[1][1]);
    Dy1 = (ele_vertices[2][1] - pos[1])/(ele_vertices[2][1] - ele_vertices[1][1]);
	//std::cout<<"dx0:"<<Dx0<<' '<<Dx1<<' '<<Dy0<<' '<<Dy1<<std::endl;
    weights[0] = Dx1 * Dy1;
    weights[1] = Dx0 * Dy1;
    weights[2] = Dx0 * Dy0;
    weights[3] = Dx1 * Dy0;
}

//explicit instantitation
template class QuadMesh<float>;
template class QuadMesh<double>;

}  //end of namespace Physika
