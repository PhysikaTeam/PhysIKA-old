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
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/matrix_base.h"
#include "Physika_Core/Vectors/vector_Nd.h"

namespace Physika{

/* class Trituple is used to store a node's message in the orthogonal list
 * every member in class Trituple is public.
 */

template <typename Scalar> class SparseMatrixIterator;
template <typename Scalar>
class Trituple
{
public:
    Trituple()
    {
        row_ = col_ = value_ = 0;
        row_next_ = NULL;
        col_next_ = NULL;
    }
    Trituple(unsigned int row, unsigned int col, Scalar value)
    {
        row_ = row;
        col_ = col;
        value_ = value;
        row_next_ = col_next_ = NULL;
    }
    bool operator==(const Trituple<Scalar> &tri2)
    {
        if(tri2.row_ != row_)return false;
        if(tri2.col_ != col_)return false;
        if(tri2.value_ != value_)return false;
        return true;
    }
    bool operator!=(const Trituple<Scalar> &tri2)
    {
        if(tri2.row_ != row_ || tri2.col_ != col_ || tri2.value_ != value_)return true;
        return false;		
    }
    unsigned int row()
    {
        return row_;
    }
    unsigned int col()
    {
        return col_;
    }
    Scalar value()
    {
        return value_;
    }
	void setRow(unsigned int i)
	{
		row_ = i;
	}
	void setCol(unsigned int j)
	{
		col_ = j;
	}
	void setValue(Scalar k)
	{
		value_ = k;
	}
private:
    unsigned int row_;
    unsigned int col_;
    Scalar value_;
    Trituple<Scalar> *row_next_;
    Trituple<Scalar> *col_next_;
};


/*class SparseMatrix is a data structure used to store SparseMatrix
 *it uses the Trituple as its node and connect them with a orthhogonal list
 */
template <typename Scalar>
class SparseMatrix: public MatrixBase
{
public:
    SparseMatrix();
    SparseMatrix(unsigned int rows, unsigned int cols);
    SparseMatrix(const SparseMatrix<Scalar> &);
    ~SparseMatrix();
    unsigned int rows() const;
    unsigned int cols() const;
    //return the number of nonZero node
    unsigned int nonZeros() const;
    // remove a node(i,j) and adjust the orthogonal list
    bool remove(unsigned int i,unsigned int j);
    //resize the SparseMatrix and data in it will be deleted
    void resize(unsigned int new_rows, unsigned int new_cols);
    SparseMatrix<Scalar> transpose() const;
    std::vector<Trituple<Scalar>> getRowElements(unsigned int ) const;
    std::vector<Trituple<Scalar>> getColElements(unsigned int ) const;
    //return value of matrix entry at index (i,j). Note: cannot be used as l-value!
    Scalar operator() (unsigned int i, unsigned int j) const;
    //insert matrix entry at index (i,j), if it already exits, replace it
    void setEntry(unsigned int i, unsigned int j, Scalar value);
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
    VectorND<Scalar> leftMultiVec(const VectorND<Scalar> &) const;
protected:
    Trituple<Scalar> * ptr(unsigned int i, unsigned int j) ;
    void allocMemory(unsigned int rows, unsigned int cols);
    void deleteRowList(Trituple<Scalar> *);
    void deleteColList(Trituple<Scalar> *);
protected:
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
//compressed orthogonal list based on Trituple
    unsigned int rows_;
    unsigned int cols_;
    Trituple<Scalar> ** row_head_;
    Trituple<Scalar> ** col_head_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    Eigen::SparseMatrix<Scalar> * ptr_eigen_sparse_matrix_ ;
    //typename typedef Eigen::SparseMatrix<Scalar>::InnerIterator SpareseIterator;
    friend class Physika::SparseMatrixIterator<Scalar>;
#endif
private:
    void compileTimeCheck()
    {
        //SparseMatrix<Scalar> is only defined for element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "SparseMatrix<Scalar> are only defined for integer types and floating-point types.");
    }
};

//overridding << for Trituple<Scalar>
template <typename Scalar>
std::ostream& operator<<(std::ostream &s, const Trituple<Scalar> &tri)
{
    s<<" ("<<tri.row()<<", "<<tri.col()<<", "<<tri.value()<<") ";
    return s;
}

//overridding << for SparseMatrix<Scalar>
template <typename Scalar>
std::ostream& operator<< (std::ostream &s, const SparseMatrix<Scalar> &mat)
{
    std::vector<Trituple<Scalar>> v;
    for(unsigned int i = 0; i < mat.rows(); ++i)
    {
        v = mat.getRowElements(i);
        for(unsigned int j=0;j< v.size();++j) s<<" ("<< v[j].row()<<", "<<v[j].col()<<", "<<v[j].value()<<") ";
        s<<std::endl;
    }
    return s;
}

//make * operator commutative
template <typename S, typename T>
SparseMatrix<T> operator* (S scale, const SparseMatrix<T> &mat)
{
    return mat*scale;
}

//multiply a row vector with a sparse matrix
template <typename Scalar>
VectorND<Scalar> operator*(const VectorND<Scalar> &vec, const SparseMatrix<Scalar> &mat)
{
    return mat.leftMultiVec(vec);
}

}  //end of namespace Physika


#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_H_
