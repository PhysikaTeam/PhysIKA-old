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
SparseMatrix<Scalar>::SparseMatrix(unsigned int rows, unsigned int cols)
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
void SparseMatrix<Scalar>::allocMemory(unsigned int rows, unsigned int cols)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    rows_ = rows;
    cols_ = cols;
    if(rows == 0) row_head_ = NULL;
    else row_head_ = new Trituple<Scalar>* [rows];
    if(cols == 0) col_head_ = NULL;
    else col_head_ = new Trituple<Scalar>* [cols];
    for(unsigned int i=0;i<rows_;++i)row_head_[i] = NULL;
    for(unsigned int i=0;i<cols_;++i)col_head_[i] = NULL;
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::deleteRowList(Trituple<Scalar> * head_)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(head_)
    {
        Trituple<Scalar> *pointer = head_, *last_pointer = NULL;
        while(pointer)
        {
            last_pointer = pointer;
            pointer= pointer->row_next_;
            if(last_pointer) delete last_pointer;
        }
    }
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::deleteColList(Trituple<Scalar> * head_)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(head_)
    {
        Trituple<Scalar> *pointer = head_, *last_pointer = NULL;
        while(pointer)
        {
            last_pointer = pointer;
            pointer= pointer->col_next_;
            if(last_pointer) delete last_pointer;
        }
    }
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>::~SparseMatrix()
{
    for(unsigned int i=0;i<rows_;++i)
    {
        deleteRowList(row_head_[i]);
        row_head_[i] = NULL;
    }
    for(unsigned int i=0;i<cols_;++i)col_head_[i] = NULL;
    if(row_head_)delete[] row_head_;
    row_head_ = NULL;
    if(col_head_)delete[] col_head_;
    col_head_ = NULL;
}

template <typename Scalar>
unsigned int SparseMatrix<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return rows_;
#endif
}

template <typename Scalar>
unsigned int SparseMatrix<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return cols_;
#endif
}

template <typename Scalar>
unsigned int SparseMatrix<Scalar>::nonZeros() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    unsigned int sum = 0;
    for(unsigned int i = 0 ; i < rows_; ++i)
    {
        Trituple<Scalar> * pointer = row_head_[i];
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
bool SparseMatrix<Scalar>::remove(unsigned int i,unsigned int j)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    Trituple<Scalar> *pointer = row_head_[i],*last_pointer = NULL;
    while(pointer)
    {
        if(pointer->col_ > j)return false;
        if(pointer->col_ == j)break;
        last_pointer = pointer;
        pointer = pointer->row_next_;
    }
    if(pointer == NULL) return false;
    if(last_pointer)last_pointer->row_next_ = pointer->row_next_;
    else row_head_[i] = pointer->row_next_;
    pointer = col_head_[j];
    last_pointer = NULL;
    while(pointer)
    {
        if(pointer->row_ > i) return false;
        if(pointer->row_ == i) break;
        last_pointer = pointer;
        pointer = pointer->col_next_;
    }
    if(pointer == NULL) return false;
    if(last_pointer)last_pointer->col_next_ = pointer->col_next_;
    else col_head_[j] = pointer->col_next_;
    delete pointer;
    return true;
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::resize(unsigned int new_rows, unsigned int new_cols)
{
    PHYSIKA_ASSERT(new_rows>=0&&new_cols>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for(unsigned int i=0;i<rows_;++i)
    {
        deleteRowList(row_head_[i]);
        row_head_[i] = NULL;
    }
    for(unsigned int i=0;i<cols_;++i)col_head_[i] = NULL;
    if(row_head_)delete[] row_head_;
    row_head_ = NULL;
    if(col_head_)delete[] col_head_;
    col_head_ = NULL;
    allocMemory(new_rows,new_cols);
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::transpose()
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<Trituple<Scalar>*> r,c;
    for(unsigned int i=0;i<rows_;++i)r.push_back(row_head_[i]);
    for(unsigned int i=0;i<cols_;++i)c.push_back(col_head_[i]);
    if(row_head_)delete[] row_head_;
    row_head_ = new Trituple<Scalar>* [cols_];
    if(col_head_)delete[] col_head_;
    col_head_ = new Trituple<Scalar>* [rows_];
    for(unsigned int i=0;i<cols_;++i)row_head_[i] = c[i];
    for(unsigned int i=0;i<rows_;++i)col_head_[i] = r[i];
    //swap rows_ and cols_
    unsigned int temp = cols_;
    cols_ = rows_;
    rows_ = temp;
    for(unsigned int i=0;i<cols_;++i)
    {
        Trituple<Scalar> *pointer = col_head_[i], *last_pointer = NULL;
        while(pointer)
        {
            //swap x row_ and col_ in every node
            unsigned int temp = pointer->col_;
            pointer->col_ = pointer->row_;
            pointer->row_ = temp;
            //swap row_next_ and col_next_ in every node
            Trituple<Scalar> *next = pointer->row_next_;
            pointer->row_next_ = pointer->col_next_;
            pointer->col_next_ = next;
            pointer = pointer->col_next_;
        }
    }
#endif
}

template <typename Scalar>
std::vector<Trituple<Scalar>> SparseMatrix<Scalar>::getRowElements(unsigned int i) const
{
    PHYSIKA_ASSERT(i>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<Trituple<Scalar>> v;
    Trituple<Scalar>* pointer = row_head_[i];
    while(pointer)
    {
        v.push_back(*pointer);
        pointer = pointer->row_next_;
    }
    return v;
#endif
}

template <typename Scalar>
std::vector<Trituple<Scalar>> SparseMatrix<Scalar>::getColElements(unsigned int i) const
{
    PHYSIKA_ASSERT(i>=0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<Trituple<Scalar>> v;
    Trituple<Scalar>* pointer = col_head_[i];
    while(pointer)
    {
        v.push_back(*pointer);
        pointer = pointer->col_next_;
    }
    return v;
#endif
}

template <typename Scalar>
Scalar SparseMatrix<Scalar>::operator() (unsigned int i, unsigned int j) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i>=0&&i<rows_);
    PHYSIKA_ASSERT(j>=0&&j<cols_);
    Trituple<Scalar>* pointer = row_head_[i];
    while(pointer)
    {
        if(pointer->col_ == j)return pointer->value_;
        pointer = pointer->row_next_;
    }
    return 0;//if is not non-zero entry, return 0
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::setEntry(unsigned int i,unsigned int j, Scalar value)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i>=0&&i<rows_);
    PHYSIKA_ASSERT(j>=0&&j<cols_);
    bool existing_entry = false;
    Trituple<Scalar> *pointer = row_head_[i],*last_pointer = NULL;
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
                if(!remove(i,j))std::cerr<<"delete pounsigned int (i,j) fail"<<std::endl;
                std::cout<<"remove over"<<std::endl;
                existing_entry = true;
                break;
            }
        }
        last_pointer = pointer;
        pointer = pointer->row_next_;
    }
    if(!existing_entry && value != 0)
    {
        Trituple<Scalar> *new_node = new Trituple<Scalar>(i,j,value);
        if(last_pointer == NULL)
        {
            row_head_[i] = new_node;
            new_node->row_next_ = pointer;
        }
        else
        {
            last_pointer->row_next_ = new_node;
            new_node->row_next_ = pointer;
        }
        pointer = col_head_[j];
        last_pointer = NULL;
        while(pointer)
        {
            if(pointer->row_ > i)break;
            last_pointer = pointer;
            pointer = pointer->col_next_;
        }
        if(last_pointer == NULL)
        {
            col_head_[j] = new_node;
            new_node->col_next_ = pointer;
        }
        else
        {
            last_pointer->col_next_ = new_node;
            new_node->col_next_ = pointer;
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
    for(unsigned int i=0; i<rows_; ++i)
    {
        Trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            unsigned int row = pointer->row_, col = pointer->col_;
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
    for(unsigned int i=0; i<mat2.rows_; ++i)
    {
        Trituple<Scalar> *pointer = mat2.row_head_[i];
        while(pointer)
        {
            unsigned int row = pointer->row_, col = pointer->col_;
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
    for(unsigned int i=0; i<rows_; ++i)
    {
        Trituple<Scalar> *pointer = row_head_[i];
        while(pointer)
        {
            unsigned int row = pointer->row_, col = pointer->col_;
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
    for(unsigned int i=0; i<mat2.rows_; ++i)
    {
        Trituple<Scalar> *pointer = mat2.row_head_[i];
        while(pointer)
        {
            unsigned int row = pointer->row_, col = pointer->col_;
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
    for(unsigned int i=0; i<mat2.rows_; ++i)
    {
        Trituple<Scalar> *pointer = mat2.row_head_[i];
        while(pointer)
        {
            unsigned int row = pointer->row_, col = pointer->col_;
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
    for(unsigned int i=0; i<rows_; ++i)
    {
        Trituple<Scalar> *pointer1 = row_head_[i], *pointer2 = mat2.row_head_[i];
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
    for(unsigned int i=0;i<rows_;++i)
    {
        Trituple<Scalar> *pointer = result.row_head_[i];
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
    for(unsigned int i=0;i<rows_;++i)
        for(unsigned int j=0;j<cols_;++j)
        {
            Scalar sum = 0;
            Trituple<Scalar> *pointer_x = row_head_[i], *pointer_y = mat2.col_head_[j];
            while(pointer_x && pointer_y)
            {
                if(pointer_x->col_ == pointer_y->row_)
                {
                    sum += pointer_x->value_ * pointer_y->value_;
                    pointer_x = pointer_x->row_next_;
                    pointer_y = pointer_y->col_next_;
                }
                else if(pointer_x->col_ < pointer_y->row_)
                {
                    pointer_x = pointer_x->row_next_;
                }
                else pointer_y = pointer_y->col_next_;
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
    for(unsigned int i=0;i<rows_;++i)
    {
        Scalar sum = 0;
        Trituple<Scalar> *pointer = row_head_[i];
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
    for(unsigned int i=0;i<rows_;++i)
    {
        Trituple<Scalar> *pointer = row_head_[i];
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
    for(unsigned int i=0;i<rows_;++i)
    {
        Trituple<Scalar> *pointer = result.row_head_[i];
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
    for(unsigned int i=0;i<rows_;++i)
    {
        Trituple<Scalar> *pointer = row_head_[i];
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
