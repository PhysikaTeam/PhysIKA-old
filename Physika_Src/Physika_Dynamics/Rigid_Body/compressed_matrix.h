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

#ifndef PHYSIKA_DYNAMICS_RIGID_BODY_COMPRESSED_MATRIX_H_
#define PHYSIKA_DYNAMICS_RIGID_BODY_COMPRESSED_MATRIX_H_

#include "Physika_Core/Vectors/vector_1d.h"
#include "Physika_Core/Vectors/vector_Nd.h"
#include "Physika_Dynamics/Rigid_Body/rotation_dof.h"

namespace Physika{

//template <typename Scalar, int Dim> class Vector<Scalar, Dim>;
//template <typename Scalar> class VectorND<Scalar>;

//The Jacobian matrix is compressed by its column.
//Before compression, a Jacobian matrix is (m, 6*n) dimension in 3d rigid body simulation where m is the number of contact points and n is the number of bodies.
//After compression, it becomes (m, 12) dimension.
//num_row_element corresponds to m, and num_column_element corresponds to n. Therefore for 3d cases the original matrix is (num_row_element, 6*num_column_element) instead of (num_row_element, num_column_element).
template <typename Scalar, int Dim>
class CompressedJacobianMatrix
{   
public:
    CompressedJacobianMatrix();
    CompressedJacobianMatrix(const CompressedJacobianMatrix<Scalar, Dim>& matrix);
    CompressedJacobianMatrix(unsigned int num_row_element, unsigned int num_column_element);
    ~CompressedJacobianMatrix();

    unsigned int rows() const;
    unsigned int cols() const;
    bool isTransposed() const;
    void setFirstValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation);
    void setSecondValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation);
    const VectorND<Scalar>& firstValue(unsigned int row_index) const;//first value of the row (or the column if it is transposed). the return value should have the dimension of 6(3d) or 4(2d)
    const VectorND<Scalar>& secondValue(unsigned int row_index) const;//second value of the row (or the column if it is transposed). the return value should have the dimension of 6(3d) or 4(2d)
    unsigned int firstColumnIndex(unsigned int row_index) const;
    unsigned int secondColumnIndex(unsigned int row_index) const;
    void buildRelationTable();
    void relatedRows(unsigned int row_index, std::vector<unsigned int>& related_index);//given a row index, get the rows containing one or more same column index with it, including the row itself

    void resize(unsigned int num_row_element, unsigned int num_column_element);
    CompressedJacobianMatrix<Scalar, Dim> transpose() const;

    CompressedJacobianMatrix<Scalar, Dim>& operator= (const CompressedJacobianMatrix<Scalar, Dim>& matrix);

protected:
    bool is_transposed_;
    unsigned int num_row_element_;
    unsigned int num_column_element_;
    std::vector<std::pair<VectorND<Scalar>, VectorND<Scalar> > > compressed_matrix_;
    std::vector<std::pair<unsigned int, unsigned int> > index_map_;//the i-th element represents the pair of two objects forming the i-th contact point
    std::vector<std::vector<unsigned int> > inverted_index_map_;//the i-th element represents the contact points on the i-th object
    std::vector<std::vector<unsigned int> > relation_table_;//the i-th element represents the contact points related to the i-th contact point
};


template <typename Scalar, int Dim>
class CompressedInertiaMatrix
{
public:
    CompressedInertiaMatrix();
    CompressedInertiaMatrix(const CompressedInertiaMatrix<Scalar, Dim>& matrix);
    CompressedInertiaMatrix(unsigned int num_element);
    ~CompressedInertiaMatrix();

    unsigned int numElement() const;
    void setMass(unsigned int index, Scalar mass);
    void setInertiaTensor(unsigned int index, SquareMatrix<Scalar, RotationDof<Dim>::degree> inertia_tensor);
    VectorND<Scalar> value(unsigned int index, unsigned int inner_index) const;//inner_index shoud in [0, Dim +  RotationDof<Dim>::degree - 1]

    void resize(unsigned int num_element);

    CompressedInertiaMatrix<Scalar, Dim>& operator= (const CompressedInertiaMatrix<Scalar, Dim>& matrix);

protected:
    unsigned int num_element_;
    std::vector<VectorND<Scalar> > compressed_matrix_;
};


template <typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim> operator* (const CompressedJacobianMatrix<Scalar, Dim>& jacobian, const CompressedInertiaMatrix<Scalar, Dim>& inertia)
{
    if(jacobian.isTransposed())
    {
        std::cerr<<"The Jacobian matrix is transposed unexpectedly!"<<std::endl;
        return jacobian;
    }
    if(jacobian.cols() != inertia.numElement())
    {
        std::cerr<<"Jacobian matrix doesn't match the size of inertia matrix!"<<std::endl;
        return jacobian;
    }
    CompressedJacobianMatrix<Scalar, Dim> result(jacobian);
    Vector<Scalar, Dim> value_translation;
    Vector<Scalar, RotationDof<Dim>::degree> value_rotation;
    for(unsigned int row_index = 0; row_index < jacobian.rows(); ++row_index)
    {
        unsigned int first_column_index = jacobian.firstColumnIndex(row_index);
        unsigned int second_column_index = jacobian.secondColumnIndex(row_index);
        for(unsigned int i = 0; i < Dim; ++i)
        {
            value_translation[i] = jacobian.firstValue(row_index).dot(inertia.value(first_column_index, i));
        }
        for(unsigned int i = 0; i < RotationDof<Dim>::degree; ++i)
        {
            value_rotation[i] = jacobian.firstValue(row_index).dot(inertia.value(first_column_index, Dim + i));
        }
        result.setFirstValue(row_index, first_column_index, value_translation, value_rotation);
        for(unsigned int i = 0; i < Dim; ++i)
        {
            value_translation[i] = jacobian.secondValue(row_index).dot(inertia.value(second_column_index, i));
        }
        for(unsigned int i = 0; i < RotationDof<Dim>::degree; ++i)
        {
            value_rotation[i] = jacobian.secondValue(row_index).dot(inertia.value(second_column_index, Dim + i));
        }
        result.setSecondValue(row_index, second_column_index, value_translation, value_rotation);
    }
    
    return result;
}
template <typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim> operator* (const CompressedInertiaMatrix<Scalar, Dim>& inertia, const CompressedJacobianMatrix<Scalar, Dim>& jacobian)
{
    return (jacobian.transpose() * inertia).transpose();
}

template <typename Scalar, int Dim>
VectorND<Scalar> operator* (const CompressedJacobianMatrix<Scalar, Dim>& jacobian, const VectorND<Scalar> vector)
{
    if(jacobian.isTransposed())
    {
        if(jacobian.rows() != vector.dims())
        {
            std::cerr<<"Jacobian matrix doesn't match the size of vector!"<<std::endl;
            return vector;
        }

        VectorND<Scalar> result((Dim + RotationDof<Dim>::degree) * jacobian.cols());
        result *= 0;
        VectorND<Scalar> element(Dim + RotationDof<Dim>::degree);
        for(unsigned int row_index = 0; row_index < jacobian.rows(); ++row_index)
        {
            unsigned int first_column_index = jacobian.firstColumnIndex(row_index);
            unsigned int second_column_index = jacobian.secondColumnIndex(row_index);
            element = jacobian.firstValue(row_index);
            for(unsigned int i = 0; i < element.dims(); ++i)
            {
                result[(Dim + RotationDof<Dim>::degree) * first_column_index + i] += element[i] * vector[row_index];
            }
            element = jacobian.secondValue(row_index);
            for(unsigned int i = 0; i < element.dims(); ++i)
            {
                result[(Dim + RotationDof<Dim>::degree) * second_column_index + i] += element[i] * vector[row_index];
            }
        }
        return result;
    }
    else
    {
        if(jacobian.cols() * (Dim + RotationDof<Dim>::degree) != vector.dims())
        {
            std::cerr<<"Jacobian matrix doesn't match the size of vector!"<<std::endl;
            return vector;
        }
        std::vector<VectorND<Scalar> > vector_divided;
        VectorND<Scalar> element(Dim + RotationDof<Dim>::degree);
        for(unsigned int i = 0; i < jacobian.cols(); ++i)
        {
            for(unsigned int j = 0; j < Dim + RotationDof<Dim>::degree; ++j)
            {
                element[j] = vector[i * (Dim + RotationDof<Dim>::degree) + j];
            }
            vector_divided.push_back(element);
        }
        VectorND<Scalar> result(jacobian.rows());
        result *= 0;
        for(unsigned int row_index = 0; row_index < jacobian.rows(); ++row_index)
        {
            unsigned int first_column_index = jacobian.firstColumnIndex(row_index);
            unsigned int second_column_index = jacobian.secondColumnIndex(row_index);
            result[row_index] = jacobian.firstValue(row_index).dot(vector_divided[first_column_index]) + jacobian.secondValue(row_index).dot(vector_divided[second_column_index]);
        }
        return result;
    }
}

template <typename Scalar, int Dim>
VectorND<Scalar> operator* (const CompressedInertiaMatrix<Scalar, Dim>& inertia, const VectorND<Scalar> vector)
{
    int dof = Dim + RotationDof<Dim>::degree;
    if(inertia.numElement() * dof != vector.dims())
    {
        std::cerr<<"Inertia matrix doesn't match the size of vector!"<<std::endl;
        return vector;
    }
    VectorND<Scalar> result(vector);
    VectorND<Scalar> element(dof);
    for(unsigned int i = 0; i < inertia.numElement(); ++i)
    {
        for(unsigned int j = 0; j < dof; ++j)
        {
            element[j] = vector[i * dof + j];
        }
        for(unsigned int j = 0; j < dof; ++j)
        {
            result[i * dof + j] = inertia.value(i, j).dot(element);
        }
    }
    return result;
}

template <typename Scalar, int Dim>
Scalar productValue(const CompressedJacobianMatrix<Scalar, Dim>& lhs, const CompressedJacobianMatrix<Scalar, Dim>& rhs, unsigned int row, unsigned int column)
{
    if(lhs.isTransposed() || !rhs.isTransposed())
    {
        std::cerr<<"The Jacobian matrix is transposed unexpectedly!"<<std::endl;
        return 0;
    }
    if(lhs.cols() != rhs.cols())
    {
        std::cerr<<"Jacobian matrix's size doesn't match!"<<std::endl;
        return 0;
    }
    if(row >= lhs.rows() || column >= rhs.rows())
    {
        std::cerr<<"Jacobian matrix's index out of range!"<<std::endl;
        return 0;
    }
    unsigned int first_column_index_lhs = lhs.firstColumnIndex(row);
    unsigned int second_column_index_lhs = lhs.secondColumnIndex(row);
    unsigned int first_column_index_rhs = rhs.firstColumnIndex(column);
    unsigned int second_column_index_rhs = rhs.secondColumnIndex(column);

    VectorND<Scalar> test1(6);
    VectorND<Scalar> test2(6);
    Scalar result = 0;
    if(first_column_index_lhs == first_column_index_rhs)
    {
        result += lhs.firstValue(row).dot(rhs.firstValue(column));
    }
    if(first_column_index_lhs == second_column_index_rhs)
    {
        result += lhs.firstValue(row).dot(rhs.secondValue(column));
    }
    if(second_column_index_lhs == first_column_index_rhs)
    {
        result += lhs.secondValue(row).dot(rhs.firstValue(column));
    }
    if(second_column_index_lhs == second_column_index_rhs)
    {
        result += lhs.secondValue(row).dot(rhs.secondValue(column));
    }
    return result;
}

}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_COMPRESSED_MATRIX_H_