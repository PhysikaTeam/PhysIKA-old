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

#include <cmath>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Arrays/array.h"
#include "Physika_Geometry/Volumetric_Meshes/tet_mesh.h"

namespace Physika{

template <typename Scalar>
TetMesh<Scalar>::TetMesh():VolumetricMesh<Scalar,3>()
{
    (this->vert_per_ele_).clear();
    (this->vert_per_ele_).push_back(4);
}

template <typename Scalar>
TetMesh<Scalar>::TetMesh(unsigned int vert_num, const Scalar *vertices, unsigned int ele_num, const unsigned int *elements)
    :VolumetricMesh<Scalar,3>(vert_num,vertices,ele_num,elements,4)
{
}

template <typename Scalar>
TetMesh<Scalar>::TetMesh(const TetMesh<Scalar> &tet_mesh)
    :VolumetricMesh<Scalar,3>(tet_mesh)
{
}

template <typename Scalar>
TetMesh<Scalar>::~TetMesh()
{
}

template <typename Scalar>
TetMesh<Scalar>& TetMesh<Scalar>::operator= (const TetMesh<Scalar> &tet_mesh)
{
    VolumetricMesh<Scalar,3>::operator= (tet_mesh);
    return *this;
}

template <typename Scalar>
void TetMesh<Scalar>::printInfo() const
{
    std::cout<<"Tetrahedral Mesh."<<std::endl;
}

template <typename Scalar>
VolumetricMeshInternal::ElementType TetMesh<Scalar>::elementType() const
{
    return VolumetricMeshInternal::TET;
}

template <typename Scalar>
unsigned int TetMesh<Scalar>::eleVertNum() const
{
    return 4;
}

template <typename Scalar>
Scalar TetMesh<Scalar>::eleVolume(unsigned int ele_idx) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"TetMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Array< Vector<Scalar,3> > ele_vertices(4);
    for(unsigned int i = 0; i < 4; ++i)
        ele_vertices[i] = this->eleVertPos(ele_idx,i);
    //volume = 1/6*|<(a-d),(b-d)x(c-d)>|
    Vector<Scalar,3> a_minus_d = ele_vertices[0] - ele_vertices[3];
    Vector<Scalar,3> b_minus_d = ele_vertices[1] - ele_vertices[3];
    Vector<Scalar,3> c_minus_d = ele_vertices[2] - ele_vertices[3]; 
    return 1.0/6*fabs(a_minus_d.dot(b_minus_d.cross(c_minus_d)));
}

template <typename Scalar>
bool TetMesh<Scalar>::containsVertex(unsigned int ele_idx, const Vector<Scalar,3> &pos) const
{
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"TetMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    std::vector<Scalar> weights;
    interpolationWeights(ele_idx,pos,weights);
    bool vert_in_ele = (weights[0]>=0)&&(weights[1]>=0)&&(weights[2]>=0)&&(weights[3]>=0);
    return vert_in_ele;
}

template <typename Scalar>
void TetMesh<Scalar>::interpolationWeights(unsigned int ele_idx, const Vector<Scalar,3> &pos, std::vector<Scalar> &weights) const
{
/*
       |x1 y1 z1 1|
  D0 = |x2 y2 z2 1|
       |x3 y3 z3 1|
       |x4 y4 z4 1|

       |x  y  z  1|
  D1 = |x2 y2 z2 1|
       |x3 y3 z3 1|
       |x4 y4 z4 1|

       |x1 y1 z1 1|
  D2 = |x  y  z  1|
       |x3 y3 z3 1|
       |x4 y4 z4 1|

       |x1 y1 z1 1|
  D3 = |x2 y2 z2 1|
       |x  y  z  1|
       |x4 y4 z4 1|

       |x1 y1 z1 1|
  D4 = |x2 y2 z2 1|
       |x3 y3 z3 1|
       |x  y  z  1|

  wi = Di / D0
*/
    if((ele_idx<0) || (ele_idx>=this->ele_num_))
    {
        std::cerr<<"TetMesh element index out of range!\n";
        std::exit(EXIT_FAILURE);
    }
    Array< Vector<Scalar,3> > ele_vertices(4);
    for(unsigned int i = 0; i < 4; ++i)
        ele_vertices[i] = this->eleVertPos(ele_idx,i);
    Scalar D[5];
    D[0] = getTetDeterminant(ele_vertices[0],ele_vertices[1],ele_vertices[2],ele_vertices[3]);
    Array< Vector<Scalar,3> > buffer(ele_vertices);
    weights.resize(4);
    for(unsigned int i = 1; i <=4; ++i)
    {
        buffer = ele_vertices;
        buffer[i-1] = pos;
        D[i] = getTetDeterminant(buffer[0],buffer[1],buffer[2],buffer[3]);
        weights[i-1] = D[i]/D[0];
    }
}

template<typename Scalar>
Scalar TetMesh<Scalar>::getTetDeterminant(const Vector<Scalar,3> &a, const Vector<Scalar,3> &b, const Vector<Scalar,3> &c, const Vector<Scalar,3> &d) const
{
    // computes the determinant of the 4x4 matrix
    // [ a 1 ]
    // [ b 1 ]
    // [ c 1 ]
    // [ d 1 ]
    SquareMatrix<Scalar,3> m0(b,c,d);
    SquareMatrix<Scalar,3> m1(a,c,d);
    SquareMatrix<Scalar,3> m2(a,b,d);
    SquareMatrix<Scalar,3> m3(a,b,c);
    return m0.determinant()*(-1) + m1.determinant() - m2.determinant() + m3.determinant();
}

//explicit instantitation
template class TetMesh<float>;
template class TetMesh<double>;

}//end of namespace Physika
