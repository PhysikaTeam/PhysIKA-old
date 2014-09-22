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
    void setFirstValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation);
    void setSecondValue(unsigned int row_index, unsigned int column_index, const Vector<Scalar, Dim>& value_translation, const Vector<Scalar, RotationDof<Dim>::degree>& value_rotation);
    VectorND<Scalar> firstValue (unsigned int row_index) const;//first value of the row (or the column if it is transposed). the return value should have the dimension of 6(3d) or 4(2d)
    VectorND<Scalar> secondValue (unsigned int row_index) const;//second value of the row (or the column if it is transposed). the return value should have the dimension of 6(3d) or 4(2d)

    void resize(unsigned int num_row_element, unsigned int num_column_element);
    CompressedJacobianMatrix<Scalar, Dim> transpose() const;

    CompressedJacobianMatrix<Scalar, Dim>& operator= (const CompressedJacobianMatrix<Scalar, Dim>& matrix);

protected:
    bool is_transposed_;
    unsigned int num_row_element_;
    unsigned int num_columns_element_;
    std::vector<std::pair<VectorND<Scalar>, VectorND<Scalar> > > compressed_matrix_;
    std::vector<std::pair<unsigned int, unsigned int> > index_map_;
};


template <typename Scalar, int Dim>
class CompressedInertiaMatrix
{
public:
    CompressedInertiaMatrix();
    ~CompressedInertiaMatrix();

    unsigned int numElement() const;
    void setMass(unsigned int index, Scalar mass);
    void setInertiaTensor(unsigned int index, SquareMatrix<Scalar, RotationDof<Dim>::degree> inertia_tensor);
    VectorND<Scalar> value(unsigned int index, unsigned int inner_index) const;//inner_index shoud in [0, Dim +  RotationDof<Dim>::degree - 1]

    void resize(unsigned int num_element);
protected:
    unsigned int num_element_;
    std::vector<VectorND<Scalar> > compressed_matrix_;
};

/*
template <typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim> operator* (const CompressedJacobianMatrix& jacobian, const CompressedInertiaMatrix& inertia)
{

}

template <typename Scalar, int Dim>
CompressedJacobianMatrix<Scalar, Dim> operator* (const CompressedInertiaMatrix& inertia, const CompressedJacobianMatrix& jacobian)
{
    
}

template <typename Scalar, int Dim>
Scalar productValue(const CompressedJacobianMatrix& lhs, const CompressedJacobianMatrix& rhs, unsigned int row, unsigned int column)
{


}*/

}

#endif //PHYSIKA_DYNAMICS_RIGID_BODY_COMPRESSED_MATRIX_H_