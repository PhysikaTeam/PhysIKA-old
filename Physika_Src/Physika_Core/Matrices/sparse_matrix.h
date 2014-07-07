/*
 * @file sparse_matrix.h 
 * @brief Definition of sparse matrix, size of the matrix is dynamic.
 * @author Fei Zhu, Liyou Xu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
#include <vector>
#include <iostream>
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Matrices/matrix_base.h"
#include "Physika_Core/Vectors/vector_Nd.h"

namespace Physika{

template <typename Scalar>
class trituple
{
public:
    trituple()
    {
        row_ = col_ = value_ = 0;
        row_next_ = NULL;
        col_next_ = NULL;
    }
    trituple(int row, int col, Scalar value)
    {
        row_ = row;
        col_ = col;
        value_ = value;
        row_next_ = col_next_ = NULL;
    }
    bool operator==(const trituple<Scalar> &tri2)
    {
        if(tri2.row_ != row_)return false;
        if(tri2.col_ != col_)return false;
        if(tri2.value_ != value_)return false;
        return true;
    }
    bool operator!=(const trituple<Scalar> &tri2)
    {
        if(tri2.row_ != row_ || tri2.col_ != col_ || tri2.value_ != value_)return true;
        return false;		
    }
public:
    int row_;
    int col_;
    Scalar value_;
    trituple<Scalar> *row_next_;
    trituple<Scalar> *col_next_;
};

template <typename Scalar>
class SparseMatrix: public MatrixBase
{
public:
    SparseMatrix();
    SparseMatrix(int rows, int cols);
    SparseMatrix(const SparseMatrix<Scalar> &);
    ~SparseMatrix();
    int rows() const;
    int cols() const;
    int nonZeros() const;
    bool remove(int i,int j);  //remove a entry in (i,j)
    void resize(int new_rows, int new_cols);
    std::vector<trituple<Scalar>> getRowElements(int ) const;
    std::vector<trituple<Scalar>> getColElements(int ) const;
    Scalar operator() (int i, int j) const;//return value of matrix entry at index (i,j). Note: cannot be used as l-value!
    void setEntry(int i, int j, Scalar value);//insert matrix entry at index (i,j), if it already exits, replace it
    SparseMatrix<Scalar> operator+ (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar>& operator+= (const SparseMatrix<Scalar> &);
    SparseMatrix<Scalar> operator- (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar>& operator-= (const SparseMatrix<Scalar> &);
    SparseMatrix<Scalar>& operator= (const SparseMatrix<Scalar> &);
    bool operator== (const SparseMatrix<Scalar> &) const;
    bool operator!= (const SparseMatrix<Scalar> &) const;
    SparseMatrix<Scalar> operator* (Scalar) const;
    SparseMatrix<Scalar> operator* (const SparseMatrix<Scalar> &) const;
    VectorND<Scalar> operator* (const VectorND<Scalar> &) const;
    SparseMatrix<Scalar>& operator*= (Scalar);
    SparseMatrix<Scalar> operator/ (Scalar) const;
    SparseMatrix<Scalar>& operator/= (Scalar);
protected:
    void allocMemory(int rows, int cols);
    void deleteRowList(trituple<Scalar> *);
    void deleteColList(trituple<Scalar> *);
protected:
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
//compressed orthogonal list based on trituple
    int rows_;
    int cols_;
    trituple<Scalar> ** row_head_;
    trituple<Scalar> ** col_head_;
#endif
};

template <typename Scalar>
std::ostream& operator<<(std::ostream &s, const trituple<Scalar> &tri)
{
    s<<" ("<<tri.row_<<", "<<tri.col_<<", "<<tri.value_<<") ";
    return s;
}

//overridding << for SparseMatrix<Scalar>
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const SparseMatrix<Scalar> &mat)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<trituple<Scalar>> v;
    for(int i = 0; i < mat.rows(); ++i)
    {
        v = mat.getRowElements(i);
        for(int j=0;j< v.size();++j) s<<" ("<< v[j].row_<<", "<<v[j].col_<<", "<<v[j].value_<<") ";
        s<<std::endl;
    }
#endif
    return s;
}

//make * operator commutative
template <typename S, typename T>
SparseMatrix<T> operator* (S scale, const SparseMatrix<T> &mat)
{
    return mat*scale;
}

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
