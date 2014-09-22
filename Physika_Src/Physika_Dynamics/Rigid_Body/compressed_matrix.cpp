/*
 * @file compressed_matrix.h 
 * @Compressed Jacobian matrix and compressed inertia matrix used in rigid body simulation
 * refer to "Iterative Dynamics with Temporal Coherence", Catto et al. 2005
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/square_matrix.h"
#include "Physika_Src/Physika_Dynamics/Rigid_Body/compressed_matrix.h"

namespace Physika{

///////////////////////////////////////////////////////////////////////////////////////
//CompressedJacobianMatrix
///////////////////////////////////////////////////////////////////////////////////////

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::CompressedJacobianMatrix():
    is_transposed_(false),
    num_row_element_(0),
    num_columns_element_(0)
{

}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::CompressedJacobianMatrix(const CompressedJacobianMatrix<Scalar, Dim>& matrix):
    is_transposed_(matrix.is_transposed_),
    num_row_element_(matrix.num_row_element_),
    num_columns_element_(matrix.num_columns_element_),
    compressed_matrix_(matrix.compressed_matrix_),
    index_map_(matrix.index_map_)
{
    
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::CompressedJacobianMatrix(unsigned int num_row_element, unsigned int num_columns_element):
    is_transposed_(false),
    num_row_element_(num_row_element),
    num_columns_element_(num_columns_element)
{
    resize(num_row_element, num_columns_element);
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::~CompressedJacobianMatrix()
{

}

template<typename Scalar, int Dim>
unsigned int CompressedJacobianMatrix<Scalar, Dim>::rows() const
{
    return num_row_element_;
}

template<typename Scalar, int Dim>
unsigned int CompressedJacobianMatrix<Scalar, Dim>::cols() const
{
    return num_columns_element_;
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::setFirstValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation)
{
    if(row_index >= num_row_element_ || column_index >= num_columns_element_)
    {
        std::cerr<<"Index out of range when setting CompressedJacobianMatrix"<<std::endl;
        return;
    }
    VectorND<Scalar> value(Dim + RotationDof<Dim>::degree);
    for(int i = 0; i < Dim; ++i)
    {
        value[i] = value_translation[i];
    }
    for(int i = 0; i < RotationDof<Dim>::degree; ++i)
    {
        value[Dim + i] = value_rotation[i];
    }
    compressed_matrix_[row_index].first = value;
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::setSecondValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation)
{
    if(row_index >= num_row_element_ || column_index >= num_columns_element_)
    {
        std::cerr<<"Index out of range when setting CompressedJacobianMatrix"<<std::endl;
        return;
    }
    VectorND<Scalar> value(Dim + RotationDof<Dim>::degree);
    for(int i = 0; i < Dim; ++i)
    {
        value[i] = value_translation[i];
    }
    for(int i = 0; i < RotationDof<Dim>::degree; ++i)
    {
        value[Dim + i] = value_rotation[i];
    }
    compressed_matrix_[row_index].second = value;
}

template<typename Scalar, int Dim>
VectorND<Scalar> CompressedJacobianMatrix<Scalar, Dim>::firstValue (unsigned int row_index) const
{
    if(row_index >= num_row_element_)
    {
        std::cerr<<"Index out of range when getting CompressedJacobianMatrix"<<std::endl;
        VectorND<Scalar> error_return;
        return error_return;
    }
    return compressed_matrix_[row_index].first;
}

template<typename Scalar, int Dim>
VectorND<Scalar> CompressedJacobianMatrix<Scalar, Dim>::secondValue (unsigned int row_index) const
{
    if(row_index >= num_row_element_)
    {
        std::cerr<<"Index out of range when getting CompressedJacobianMatrix"<<std::endl;
         VectorND<Scalar> error_return;
        return error_return;
    }
    return compressed_matrix_[row_index].second;
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::resize(unsigned int num_row_element, unsigned int num_column_element)
{

}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim> CompressedJacobianMatrix<Scalar, Dim>::transpose() const
{
    CompressedJacobianMatrix<Scalar, Dim> transposed(*this);
    transposed.is_transposed_ = true;
    return transposed;
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>& CompressedJacobianMatrix<Scalar, Dim>::operator= (const CompressedJacobianMatrix<Scalar, Dim>& matrix)
{
    CompressedJacobianMatrix<Scalar, Dim> temp_matrix(matrix);
    is_transposed_ = matrix.is_transposed_;
    num_row_element_ = matrix.num_row_element_;
    num_columns_element_ = matrix.num_columns_element_;
    compressed_matrix_.swap(temp_matrix.compressed_matrix_);
    index_map_.swap(temp_matrix.index_map_);
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////
//CompressedInertiaMatrix
///////////////////////////////////////////////////////////////////////////////////////

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>::CompressedInertiaMatrix()
{

}

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>::~CompressedInertiaMatrix()
{

}

template<typename Scalar, int Dim>
unsigned int CompressedInertiaMatrix<Scalar, Dim>::numElement() const
{
    return 0;
}

template<typename Scalar, int Dim>
void CompressedInertiaMatrix<Scalar, Dim>::setMass(unsigned int index, Scalar mass)
{

}

template<typename Scalar, int Dim>
void CompressedInertiaMatrix<Scalar, Dim>::setInertiaTensor(unsigned int index, SquareMatrix<Scalar, RotationDof<Dim>::degree> inertia_tensor)
{

}

template<typename Scalar, int Dim>
VectorND<Scalar> CompressedInertiaMatrix<Scalar, Dim>::value(unsigned int index, unsigned int inner_index) const
{
    VectorND<Scalar> a;
    return a;
}

template<typename Scalar, int Dim>
void CompressedInertiaMatrix<Scalar, Dim>::resize(unsigned int num_element)
{

}

template class CompressedJacobianMatrix<float, 2>;
template class CompressedJacobianMatrix<double, 2>;
template class CompressedJacobianMatrix<float, 3>;
template class CompressedJacobianMatrix<double, 3>;
template class CompressedInertiaMatrix<float, 2>;
template class CompressedInertiaMatrix<double, 2>;
template class CompressedInertiaMatrix<float, 3>;
template class CompressedInertiaMatrix<double, 3>;

}