/*
 * @file sparse_matrix.cpp 
 * @brief Definition of sparse matrix, size of the matrix is dynamic.
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
#include "Physika_Core/Matrices/sparse_matrix.h"

namespace Physika{

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix()
{
    allocMemory(0,0);
}

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(int rows, int cols)
{
    PHYSIKA_ASSERT(rows>=0&&cols>=0);
    allocMemory(rows,cols);
}

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(const SparseMatrix<Scalar> &mat2)
{
    allocMemory(0,0);
    *this = mat2;
}

template <typename Scalar>
void SparseMatrix<Scalar>::allocMemory(int rows, int cols)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    rows_ = rows;
    cols_ = cols;
    if(rows == 0) row_head_ = NULL;
    else row_head_ = new trituple<Scalar>* [rows];
    if(cols == 0) col_head_ = NULL;
    else col_head_ = new trituple<Scalar>* [cols];
    for(int i=0;i<rows_;++i)row_head_[i] = NULL;
    for(int i=0;i<cols_;++i)col_head_[i] = NULL;
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::deleteRowList(trituple<Scalar> * head_)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(head_)
    {
        trituple<Scalar> *pointer = head_, *lastPointer = NULL;
        while(pointer)
        {
            lastPointer = pointer;
            pointer= pointer->row_next_;
            if(lastPointer) delete lastPointer;
        }
    }
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::deleteColList(trituple<Scalar> * head_)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(head_)
    {
        trituple<Scalar> *pointer = head_, *lastPointer = NULL;
        while(pointer)
        {
            lastPointer = pointer;
            pointer= pointer->col_next_;
            if(lastPointer) delete lastPointer;
        }
    }
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>::~SparseMatrix()
{
    for(int i=0;i<rows_;++i)
    {
        deleteRowList(row_head_[i]);
        row_head_[i] = NULL;
    }
    for(int i=0;i<cols_;++i)col_head_[i] = NULL;
    if(row_head_)delete[] row_head_;
    row_head_ = NULL;
    if(col_head_)delete[] col_head_;
    col_head_ = NULL;
}

template <typename Scalar>
int SparseMatrix<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return rows_;
#endif
}

template <typename Scalar>
int SparseMatrix<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return cols_;
#endif
}

template <typename Scalar>
int SparseMatrix<Scalar>::nonZeros() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    int sum = 0;
    for(int i = 0 ; i < rows_; ++i)
    {
        trituple<Scalar> * pointer = row_head_[i];
        while(pointer)
        {
            sum++;
            pointer = pointer->row_next_;
        }
    }
    return sum;
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::remove(int i,int j)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    trituple<Scalar> *pointer = row_head_[i],*lastPointer = NULL;
    while(pointer)
    {
        if(pointer->col_ > j)return false;
        if(pointer->col_ == j)break;
        lastPointer = pointer;
        pointer = pointer->row_next_;
    }
    if(pointer == NULL) return false;
    if(lastPointer)lastPointer->row_next_ = pointer->row_next_;
    else row_head_[i] = pointer->row_next_;
    pointer = col_head_[j];
    lastPointer = NULL;
    while(pointer)
    {
        if(pointer->row_ > i) return false;
        if(pointer->row_ == i) break;
        lastPointer = pointer;
        pointer = pointer->col_next_;
    }
    if(pointer == NULL) return false;
    if(lastPointer)lastPointer->col_next_ = pointer->col_next_;
    else col_head_[j] = pointer->col_next_;
    delete pointer;
    return true;
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::resize(int new_rows, int new_cols)
{
    PHYSIKA_ASSERT(new_rows>=0&&new_cols>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for(int i=0;i<rows_;++i)
    {
        deleteRowList(row_head_[i]);
        row_head_[i] = NULL;
    }
    for(int i=0;i<cols_;++i)col_head_[i] = NULL;
    if(row_head_)delete[] row_head_;
    row_head_ = NULL;
    if(col_head_)delete[] col_head_;
    col_head_ = NULL;
    allocMemory(new_rows,new_cols);
#endif
}

template <typename Scalar>
std::vector<trituple<Scalar>> SparseMatrix<Scalar>::getRowElements(int i) const
{
    PHYSIKA_ASSERT(i>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<trituple<Scalar>> v;
    trituple<Scalar>* pointer = row_head_[i];
    while(pointer)
    {
        v.push_back(*pointer);
        pointer = pointer->row_next_;
    }
    return v;
#endif
}

template <typename Scalar>
std::vector<trituple<Scalar>> SparseMatrix<Scalar>::getColElements(int i) const
{
    PHYSIKA_ASSERT(i>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<trituple<Scalar>> v;
    trituple<Scalar>* pointer = col_head_[i];
    while(pointer)
    {
        v.push_back(*pointer);
        pointer = pointer->col_next_;
    }
    return v;
#endif
}

template <typename Scalar>
Scalar SparseMatrix<Scalar>::operator() (int i, int j) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i>=0&&i<rows_);
    PHYSIKA_ASSERT(j>=0&&j<cols_);
    trituple<Scalar>* pointer = row_head_[i];
    while(pointer)
    {
        if(pointer->col_ == j)return pointer->value_;
        pointer = pointer->row_next_;
    }
    return 0;//if is not non-zero entry, return 0
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::setEntry(int i,int j, Scalar value)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i>=0&&i<rows_);
    PHYSIKA_ASSERT(j>=0&&j<cols_);
    bool existing_entry = false;
    trituple<Scalar> *pointer = row_head_[i],*lastPointer = NULL;
    while(pointer)
    {
        if(pointer->col_ > j)break;
        if(pointer->col_ == j)
        {
            if(value != 0)
            {
                pointer->value_ = value;
                existing_entry = true;
                break;
            }
            else
            {
                std::cout<<"begin remove (i,j)"<<std::endl;
                if(!remove(i,j))std::cerr<<"delete point (i,j) fail"<<std::endl;
                std::cout<<"remove over"<<std::endl;
                existing_entry = true;
                break;
            }
        }
        lastPointer = pointer;
        pointer = pointer->row_next_;
    }
    if(!existing_entry && value != 0)
    {
        trituple<Scalar> *newNode = new trituple<Scalar>(i,j,value);
        if(lastPointer == NULL)
        {
            row_head_[i] = newNode;
            newNode->row_next_ = pointer;
        }
        else
        {
            lastPointer->row_next_ = newNode;
            newNode->row_next_ = pointer;
        }
        pointer = col_head_[j];
        lastPointer = NULL;
        while(pointer)
        {
            if(pointer->row_ > i)break;
            lastPointer = pointer;
            pointer = pointer->col_next_;
        }
        if(lastPointer == NULL)
        {
            col_head_[j] = newNode;
            newNode->col_next_ = pointer;
        }
        else
        {
            lastPointer->col_next_ = newNode;
            newNode->col_next_ = pointer;
        }
    }
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator+ (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    SparseMatrix<Scalar> result(mat2);
    for(int i=0; i<rows_; ++i)
    {
        trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            int row = pointer->row_, col = pointer->col_;
            result.setEntry(row, col, pointer->value_+result(row, col));
            pointer = pointer->row_next_;
        }
    }
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator+= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    for(int i=0; i<mat2.rows_; ++i)
    {
        trituple<Scalar> *pointer = mat2.row_head_[i];
        while(pointer)
        {
            int row = pointer->row_, col = pointer->col_;
            this->setEntry(row, col, pointer->value_+(*this)(row, col));
            pointer = pointer->row_next_;
        }        
    }
    return *this;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator- (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    SparseMatrix<Scalar> result(mat2);
    for(int i=0; i<rows_; ++i)
    {
        trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            int row = pointer->row_, col = pointer->col_;
            result.setEntry(row, col, pointer->value_ - result(row, col));
            pointer = pointer->row_next_;
        }
    }
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator-= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    for(int i=0; i<mat2.rows_; ++i)
    {
        trituple<Scalar> *pointer = mat2.row_head_[i];
        while(pointer)
        {
            int row = pointer->row_, col = pointer->col_;
            this->setEntry(row, col, (*this)(row, col) - pointer->value_);
            pointer = pointer->row_next_;
        }        
    }
    return *this;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    this->resize(mat2.rows(),mat2.cols());
    for(int i=0; i<mat2.rows_; ++i)
    {
        trituple<Scalar> *pointer = mat2.row_head_[i];
        while(pointer)
        {
            int row = pointer->row_, col = pointer->col_;
            this->setEntry(row, col,pointer->value_);
            pointer = pointer->row_next_;
        }        
    }
    return *this;
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::operator== (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(rows_ != mat2.rows() || cols_ != mat2.cols())return false;
    for(int i=0; i<rows_; ++i)
    {
        trituple<Scalar> *pointer1 = row_head_[i], *pointer2 = mat2.row_head_[i];
        while(pointer1)
        {
            if(*pointer1 != *pointer2)return false;
            pointer1 = pointer1->row_next_;
            pointer2 = pointer2->row_next_;
        } 
        if(pointer2)return false;
    }
    return true;
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::operator!= (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(*this == mat2)return false;
    else return true;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator* (Scalar scale) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    SparseMatrix<Scalar> result(*this);
    for(int i=0;i<rows_;++i)
    {
        trituple<Scalar> *pointer = result.row_head_[i];
        while(pointer)
        {
            pointer->value_ *= scale;
            pointer = pointer->row_next_;
        }
    }
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator* (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(cols_ != mat2.rows_)
    {
        std::cerr<<"operator * between two SparseMatrixes failed because they don't match"<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    SparseMatrix<Scalar> result(rows_,mat2.cols_);
    for(int i=0;i<rows_;++i)
        for(int j=0;j<cols_;++j)
        {
            Scalar sum = 0;
            trituple<Scalar> *pointerx = row_head_[i], *pointery = mat2.col_head_[j];
            while(pointerx && pointery)
            {
                if(pointerx->col_ == pointery->row_)
                {
                    sum += pointerx->value_ * pointery->value_;
                    pointerx = pointerx->row_next_;
                    pointery = pointery->col_next_;
                }
                else if(pointerx->col_ < pointery->row_)
                {
                    pointerx = pointerx->row_next_;
                }
                else pointery = pointery->col_next_;
            }
            if(sum) result.setEntry(i,j,sum);
        }
    return result;
#endif
}

template <typename Scalar>
VectorND<Scalar> SparseMatrix<Scalar>::operator* (const VectorND<Scalar> &vec) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(cols_ != vec.dims())
    {
        std::cerr<<"operator * between SpaseMatrix and VectorND failed because the two don't match"<<std::endl;
        std::exit(EXIT_FAILURE);
    }
    VectorND<Scalar> result(rows_, 0);
    for(int i=0;i<rows_;++i)
    {
        Scalar sum = 0;
        trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            sum += pointer->value_ * vec[pointer->row_];
            pointer = pointer->row_next_;
        }
        result[i] = sum;
    }
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator*=(Scalar scale)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for(int i=0;i<rows_;++i)
    {
        trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            pointer->value_ *= scale;
            pointer = pointer->row_next_;
        }
    }
    return *this;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator/ (Scalar scale) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    SparseMatrix<Scalar> result(*this);
    for(int i=0;i<rows_;++i)
    {
        trituple<Scalar> *pointer = result.row_head_[i];
        while(pointer)
        {
            pointer->value_ /= scale;
            pointer = pointer->row_next_;
        }
    }
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator/=(Scalar scale)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for(int i=0;i<rows_;++i)
    {
        trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            pointer->value_ /= scale;
            pointer = pointer->row_next_;
        }
    }
    return *this;
#endif
}

//explicit instantiation of template so that it could be compiled into a lib
template class SparseMatrix<unsigned char>;
template class SparseMatrix<unsigned short>;
template class SparseMatrix<unsigned int>;
template class SparseMatrix<unsigned long>;
template class SparseMatrix<unsigned long long>;
template class SparseMatrix<signed char>;
template class SparseMatrix<short>;
template class SparseMatrix<int>;
template class SparseMatrix<long>;
template class SparseMatrix<long long>;
template class SparseMatrix<float>;
template class SparseMatrix<double>;
template class SparseMatrix<long double>;
}  //end of namespace Physika
