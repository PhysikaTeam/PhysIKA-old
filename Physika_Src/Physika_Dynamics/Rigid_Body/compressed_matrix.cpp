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
#include "Physika_Core/Matrices/matrix_1x1.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Matrices/square_matrix.h"
#include "Physika_Dynamics/Rigid_Body/compressed_matrix.h"
#include <algorithm>

namespace Physika{

///////////////////////////////////////////////////////////////////////////////////////
//CompressedJacobianMatrix
///////////////////////////////////////////////////////////////////////////////////////

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::CompressedJacobianMatrix():
    is_transposed_(false),
    num_row_element_(0),
    num_column_element_(0)
{

}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::CompressedJacobianMatrix(const CompressedJacobianMatrix<Scalar, Dim>& matrix):
    is_transposed_(matrix.is_transposed_),
    num_row_element_(matrix.num_row_element_),
    num_column_element_(matrix.num_column_element_),
    compressed_matrix_(matrix.compressed_matrix_),
    index_map_(matrix.index_map_),
    inverted_index_map_(matrix.inverted_index_map_),
    relation_table_(matrix.relation_table_)
{
    
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::CompressedJacobianMatrix(unsigned int num_row_element, unsigned int num_columns_element):
    is_transposed_(false),
    num_row_element_(num_row_element),
    num_column_element_(num_columns_element)
{
    resize(num_row_element, num_columns_element);
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>::~CompressedJacobianMatrix()
{

}

template<typename Scalar, int Dim>
bool CompressedJacobianMatrix<Scalar, Dim>::isTransposed() const
{
    return is_transposed_;
}

template<typename Scalar, int Dim>
unsigned int CompressedJacobianMatrix<Scalar, Dim>::rows() const
{
    return num_row_element_;
}

template<typename Scalar, int Dim>
unsigned int CompressedJacobianMatrix<Scalar, Dim>::cols() const
{
    return num_column_element_;
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::setFirstValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation)
{
    if(row_index >= num_row_element_ || column_index >= num_column_element_)
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
    index_map_[row_index].first = column_index;
    //inverted_index_map_[column_index].push_back(row_index);
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::setSecondValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation)
{
    if(row_index >= num_row_element_ || column_index >= num_column_element_)
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
    index_map_[row_index].second = column_index;
    //inverted_index_map_[column_index].push_back(row_index);
}

template<typename Scalar, int Dim>
const VectorND<Scalar>& CompressedJacobianMatrix<Scalar, Dim>::firstValue (unsigned int row_index) const
{
    if(row_index >= num_row_element_)
    {
        std::cerr<<"Index out of range when getting CompressedJacobianMatrix's value"<<std::endl;
        VectorND<Scalar> error_return;
        return error_return;
    }
    return compressed_matrix_[row_index].first;
}

template<typename Scalar, int Dim>
const VectorND<Scalar>& CompressedJacobianMatrix<Scalar, Dim>::secondValue (unsigned int row_index) const
{
    if(row_index >= num_row_element_)
    {
        std::cerr<<"Index out of range when getting CompressedJacobianMatrix's value"<<std::endl;
        VectorND<Scalar> error_return;
        return error_return;
    }
    return compressed_matrix_[row_index].second;
}

template<typename Scalar, int Dim>
unsigned int CompressedJacobianMatrix<Scalar, Dim>::firstColumnIndex(unsigned int row_index) const
{
    if(row_index >= num_row_element_)
    {
        std::cerr<<"Index out of range when getting CompressedJacobianMatrix's index"<<std::endl;
        return 0;
    }
    return index_map_[row_index].first;
}

template<typename Scalar, int Dim>
unsigned int CompressedJacobianMatrix<Scalar, Dim>::secondColumnIndex(unsigned int row_index) const
{
    if(row_index >= num_row_element_)
    {
        std::cerr<<"Index out of range when getting CompressedJacobianMatrix's index"<<std::endl;
        return 0;
    }
    return index_map_[row_index].second;
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::buildRelationTable()
{
    //initialize inverted_index_map_
    for(unsigned int row_index = 0; row_index < num_row_element_; ++row_index)
    {
        inverted_index_map_[index_map_[row_index].first].push_back(row_index);
        inverted_index_map_[index_map_[row_index].second].push_back(row_index);
    }

    //for each row, find its related rows
    for(unsigned int row_index = 0; row_index < num_row_element_; ++row_index)
    {
        unsigned int first_column = firstColumnIndex(row_index);
        unsigned int second_column = secondColumnIndex(row_index);
        int size_first(inverted_index_map_[first_column].size()), size_second(inverted_index_map_[second_column].size());
        relation_table_[row_index].reserve(size_first + size_second);

        unsigned int pointer_first(0), pointer_second(0);
        unsigned int last_value(100000000), current_value;

        //merge two ordered arrays and ignore duplicated elements
        while(pointer_first < size_first || pointer_second < size_second)
        {
            if(pointer_first == size_first)
            {
                current_value = inverted_index_map_[second_column][pointer_second];
                pointer_second ++;
            }
            else if(pointer_second == size_second)
            {
                current_value = inverted_index_map_[first_column][pointer_first];
                pointer_first ++;
            }
            else if(inverted_index_map_[first_column][pointer_first] < inverted_index_map_[second_column][pointer_second])
            {
                current_value = inverted_index_map_[first_column][pointer_first];
                pointer_first ++;
            }
            else
            {
                current_value = inverted_index_map_[second_column][pointer_second];
                pointer_second ++;
            }
            if(current_value == last_value)
                continue;
            relation_table_[row_index].push_back(current_value);
            last_value = current_value;
        }
    }
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::relatedRows(unsigned int row_index, std::vector<unsigned int>& related_index)
{
    related_index = relation_table_[row_index];
}

template<typename Scalar, int Dim>
void CompressedJacobianMatrix<Scalar, Dim>::resize(unsigned int num_row_element, unsigned int num_column_element)
{
    is_transposed_ = false;
    num_row_element_ = num_row_element;
    num_column_element_ = num_column_element;
    compressed_matrix_.clear();
    compressed_matrix_.resize(num_row_element);
    index_map_.clear();
    index_map_.resize(num_row_element);
    inverted_index_map_.clear();
    inverted_index_map_.resize(num_column_element);
    relation_table_.clear();
    relation_table_.resize(num_row_element);
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim> CompressedJacobianMatrix<Scalar, Dim>::transpose() const
{
    CompressedJacobianMatrix<Scalar, Dim> transposed(*this);
    transposed.is_transposed_ = !this->isTransposed();
    return transposed;
}

template<typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim>& CompressedJacobianMatrix<Scalar, Dim>::operator= (const CompressedJacobianMatrix<Scalar, Dim>& matrix)
{
    CompressedJacobianMatrix<Scalar, Dim> temp_matrix(matrix);
    is_transposed_ = matrix.is_transposed_;
    num_row_element_ = matrix.num_row_element_;
    num_column_element_ = matrix.num_column_element_;
    compressed_matrix_.swap(temp_matrix.compressed_matrix_);
    index_map_.swap(temp_matrix.index_map_);
    inverted_index_map_.swap(temp_matrix.inverted_index_map_);
    relation_table_.swap(temp_matrix.relation_table_);
    return *this;
}

///////////////////////////////////////////////////////////////////////////////////////
//CompressedInertiaMatrix
///////////////////////////////////////////////////////////////////////////////////////

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>::CompressedInertiaMatrix():
    num_element_(0)
{

}

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>::CompressedInertiaMatrix(const CompressedInertiaMatrix<Scalar, Dim>& matrix):
    num_element_(matrix.num_element_),
    compressed_matrix_(matrix.compressed_matrix_)
{

}

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>::CompressedInertiaMatrix(unsigned int num_element):
    num_element_(num_element)
{
    resize(num_element);
}

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>::~CompressedInertiaMatrix()
{

}

template<typename Scalar, int Dim>
unsigned int CompressedInertiaMatrix<Scalar, Dim>::numElement() const
{
    return num_element_;
}

template<typename Scalar, int Dim>
void CompressedInertiaMatrix<Scalar, Dim>::setMass(unsigned int index, Scalar mass)
{
    if(index >= num_element_)
    {
        std::cerr<<"Index out of range when getting CompressedInertiaMatrix"<<std::endl;
        return;
    }
    VectorND<Scalar> row(Dim + RotationDof<Dim>::degree);
    for(unsigned int i = 0; i < Dim; ++i)
    {
        row *= 0;
        row[i] = mass;
        compressed_matrix_[(Dim + RotationDof<Dim>::degree) * index + i] = row;
    }
}

template<typename Scalar, int Dim>
void CompressedInertiaMatrix<Scalar, Dim>::setInertiaTensor(unsigned int index, SquareMatrix<Scalar, RotationDof<Dim>::degree> inertia_tensor)
{
    if(index >= num_element_)
    {
        std::cerr<<"Index out of range when getting CompressedInertiaMatrix"<<std::endl;
        return;
    }
    VectorND<Scalar> row(Dim + RotationDof<Dim>::degree);
    for(unsigned int i = 0; i < RotationDof<Dim>::degree; ++i)
    {
        row *= 0;
        for(unsigned int j = 0; j < RotationDof<Dim>::degree; ++j)
        {
            row[Dim + j] = inertia_tensor(i, j);
        }
        compressed_matrix_[(Dim + RotationDof<Dim>::degree) * index + Dim + i] = row;
    }
}

template<typename Scalar, int Dim>
VectorND<Scalar> CompressedInertiaMatrix<Scalar, Dim>::value(unsigned int index, unsigned int inner_index) const
{
    if(index >= num_element_ || inner_index >= (Dim + RotationDof<Dim>::degree))
    {
        std::cerr<<"Index out of range when getting CompressedInertiaMatrix"<<std::endl;
        VectorND<Scalar> error_return;
        return error_return;
    }
    return compressed_matrix_[(Dim + RotationDof<Dim>::degree) * index + inner_index];
}

template<typename Scalar, int Dim>
void CompressedInertiaMatrix<Scalar, Dim>::resize(unsigned int num_element)
{
    num_element_ = num_element;
    compressed_matrix_.resize((Dim + RotationDof<Dim>::degree) * num_element);
}

template<typename Scalar, int Dim>
CompressedInertiaMatrix<Scalar, Dim>& CompressedInertiaMatrix<Scalar, Dim>::operator= (const CompressedInertiaMatrix<Scalar, Dim>& matrix)
{
    CompressedInertiaMatrix<Scalar, Dim> temp_matrix(matrix);
    num_element_ = matrix.num_element_;
    compressed_matrix_.swap(temp_matrix.compressed_matrix_);
    return *this;
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