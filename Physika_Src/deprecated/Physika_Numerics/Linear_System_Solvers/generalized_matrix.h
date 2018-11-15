/*
 * @file generalized_matrix.h
 * @brief generalized matrix class to represent the coefficient matrix A in a linear system
 *        Ax = b. The generalized matrix hides sparse/dense information of A to provide a
 *        unified interface for LinearSystem class.
 * @author Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_GENERALIZED_MATRIX_H_
#define PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_GENERALIZED_MATRIX_H_

namespace Physika{

template <typename Scalar> class MatrixMxN;
template <typename Scalar> class SparseMatrix;

/*
 * GeneralizedMatrix: provide unified interface for LinearSystem classes
 * specific solver can call denseMatrix()/sparseMatrix() method first to get the pointer
 * to the exact content and directly operate on it.
 */

template <typename Scalar>
class GeneralizedMatrix
{
public:
    explicit GeneralizedMatrix(const MatrixMxN<Scalar> &matrix);
    explicit GeneralizedMatrix(const SparseMatrix<Scalar> &matrix);
    GeneralizedMatrix(const GeneralizedMatrix<Scalar> &matrix);
    ~GeneralizedMatrix();

    //simple operations, explicitly call methods of actual matrix for more advanced operations
    unsigned int rows() const;
    unsigned int cols() const;
    GeneralizedMatrix<Scalar>& operator= (const GeneralizedMatrix<Scalar> &matrix);
    bool operator== (const GeneralizedMatrix<Scalar> &matrix) const;
    bool operator!= (const GeneralizedMatrix<Scalar> &matrix) const;

    //get the actual content of generalized matrix, return NULL for
    //incompatible call, e.g., call denseMatrix() while the matrix is sparse
    const MatrixMxN<Scalar>* denseMatrix() const;
    MatrixMxN<Scalar>* denseMatrix();
    const SparseMatrix<Scalar>* sparseMatrix() const;
    SparseMatrix<Scalar>* sparseMatrix();
protected:
    GeneralizedMatrix(); //default constructor made protected
    void clearMatrix();
protected:
    MatrixMxN<Scalar> *dense_matrix_;
    SparseMatrix<Scalar> *sparse_matrix_;
};

}  //end of namespace Physika

#endif //PHYSIKA_NUMERICS_LINEAR_SYSTEM_SOLVERS_GENERALIZED_MATRIX_H_
