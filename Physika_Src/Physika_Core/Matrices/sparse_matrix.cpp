/*
 * @file sparse_matrix.cpp 
 * @brief Definition of sparse matrix, size of the matrix is dynamic.
 * @author Liyou Xu, Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <cstdlib>
#include <iostream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Core/Matrices/sparse_matrix.h"

namespace Physika{

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(matrix_compressed_mode priority)
{
    allocMemory(0,0,priority);
}

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(unsigned int rows, unsigned int cols, matrix_compressed_mode priority)
{
    allocMemory(rows,cols,priority);
}

template <typename Scalar>
SparseMatrix<Scalar>::SparseMatrix(const SparseMatrix<Scalar> &mat2)
{
    allocMemory(mat2.rows(),mat2.cols(),mat2.priority_);
    *this = mat2;
}

//initialize the vector line_index_
template <typename Scalar>
void SparseMatrix<Scalar>::allocMemory(unsigned int rows, unsigned int cols, matrix_compressed_mode priority)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    rows_ = rows;
    cols_ = cols;
    priority_ = priority;
    if(priority_ == ROW_MAJOR) //row-wise
    {
        for(unsigned int i=0;i<=rows_;++i)
        {
            line_index_.push_back(0);
        }
    }
    else
    {
        for (unsigned int i = 0; i <= cols_; ++i)
        {
            line_index_.push_back(0);
        }
    }
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
	if (priority_ == ROW_MAJOR)ptr_eigen_sparse_matrix_ = new Eigen::SparseMatrix<Scalar, Eigen::RowMajor>(rows, cols);
	else ptr_eigen_sparse_matrix_ = new Eigen::SparseMatrix<Scalar, Eigen::ColMajor>(rows, cols);
    PHYSIKA_ASSERT(ptr_eigen_sparse_matrix_);
#endif

}



template <typename Scalar>
SparseMatrix<Scalar>::~SparseMatrix()
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    delete ptr_eigen_sparse_matrix_;
#endif
}

template <typename Scalar>
unsigned int SparseMatrix<Scalar>::rows() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return rows_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    return (*ptr_eigen_sparse_matrix_).rows();
#endif
}

template <typename Scalar>
unsigned int SparseMatrix<Scalar>::cols() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    return cols_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    return (*ptr_eigen_sparse_matrix_).cols();
#endif
}

template <typename Scalar>
unsigned int SparseMatrix<Scalar>::nonZeros() const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    unsigned int non_zeros_ = elements_.size();
    for (unsigned int i = 0; i < elements_.size(); ++i)
    {
        if (isEqual(elements_[i].value(),static_cast<Scalar>(0)))
            non_zeros_--;
    }
    return non_zeros_;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    return (*ptr_eigen_sparse_matrix_).nonZeros();
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::remove(unsigned int i,unsigned int j)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    unsigned int k;
    if (priority_ == 0) k = i;
    else k = j;
    unsigned int begin = line_index_[k], end = line_index_[k + 1];
    for (unsigned int i1 = begin; i1 < end; ++i1)
    {
        if (elements_[i1].row() == i&&elements_[i1].col() == j)
        {
            elements_.erase(elements_.begin() + i1);
            if (priority_)
            {
                for (unsigned int j1 = j + 1; j1 <= cols_; ++j1)
                {
                    line_index_[j1] -= 1;
                }
            }
            else{
                for (unsigned int j1 = i + 1; j1 <= rows_; ++j1)
                {
                    line_index_[j1] -= 1;
                }
            }
            return true;
        }
    }
    return false;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    (*ptr_eigen_sparse_matrix_).coeffRef(i,j) = 0;
    return true;
#endif
}

template <typename Scalar>
void SparseMatrix<Scalar>::resize(unsigned int new_rows, unsigned int new_cols)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    allocMemory(new_rows,new_cols, priority_);
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    (*ptr_eigen_sparse_matrix_).resize(new_rows, new_cols);
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::transpose() const
{
    SparseMatrix<Scalar> result(this->cols(), this->rows());
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    result.elements_ = elements_;
    for (unsigned int i = 0; i < elements_.size(); ++i)
    {
        result.elements_[i].setRow(elements_[i].col());
        result.elements_[i].setCol(elements_[i].row());
    }
    result.line_index_ = line_index_;
	if (priority_ == ROW_MAJOR)result.priority_ = COL_MAJOR;
	else result.priority_ = ROW_MAJOR;
    return result;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    *result.ptr_eigen_sparse_matrix_ = (*ptr_eigen_sparse_matrix_).transpose();
    return result;
#endif
}

template <typename Scalar>
std::vector<Trituple<Scalar>> SparseMatrix<Scalar>::getRowElements(unsigned int i) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<Trituple<Scalar>> v;
    if (priority_)
    {
        for (unsigned int i1 = 0; i1 < elements_.size();++i1)
        if (elements_[i1].row() == i)v.push_back(elements_[i1]);
    }
    else{
        for(unsigned int i1=line_index_[i];i1<line_index_[i+1];++i1)v.push_back(elements_[i1]);
    }
    return v;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    std::vector<Trituple<Scalar>> v;
    for(int j = 0;j<ptr_eigen_sparse_matrix_->cols();++j)
    {
        Scalar value_temp = (*ptr_eigen_sparse_matrix_).coeff(i,j);
        if(value_temp != 0)
        {
            v.push_back(Trituple<Scalar>(i,j,value_temp));
        }
    }
    return v;
#endif
}

template <typename Scalar>
std::vector<Trituple<Scalar>> SparseMatrix<Scalar>::getColElements(unsigned int i) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    std::vector<Trituple<Scalar>> v;
    if (priority_)
    {
        for(unsigned int i1=line_index_[i];i1<line_index_[i+1];++i1)v.push_back(elements_[i1]);
    }
    else{
        for (unsigned int i1 = 0; i1 < elements_.size(); ++i1)
        if (elements_[i1].col() == i)v.push_back(elements_[i1]);
    }
    return v;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    std::vector<Trituple<Scalar>> v;
    for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(*ptr_eigen_sparse_matrix_, i); it; ++it)
    {
        v.push_back(Trituple<Scalar>(it.row(), it.col(), it.value()));
    }
    return v;
#endif
}

template <typename Scalar>
Scalar SparseMatrix<Scalar>::operator() (unsigned int i, unsigned int j) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i<rows_);
    PHYSIKA_ASSERT(j<cols_);
    if (priority_)
    {
        for (unsigned int i1 = line_index_[j]; i1 < line_index_[j + 1];++i1) if (elements_[i1].row() == i) return elements_[i1].value();
    }
    else{
        for (unsigned int i1 = line_index_[i]; i1 < line_index_[i + 1]; ++i1) if (elements_[i1].col() == j)return elements_[i1].value();
    }
    return 0;//if is not non-zero entry, return 0
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    return (*ptr_eigen_sparse_matrix_).coeff(i,j);
#endif
}


template <typename Scalar>
void SparseMatrix<Scalar>::setEntry(unsigned int i,unsigned int j, Scalar value)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(i<rows_);
    PHYSIKA_ASSERT(j<cols_);
    if (priority_)                        //col-wise
    {
        if (line_index_[j] == line_index_[j + 1])            //if this is a empty column
        {
            unsigned int i1 = line_index_[j];
            elements_.insert(elements_.begin() + i1, Trituple<Scalar>(i, j, value));
            for (unsigned int i2 = j + 1; i2 <= cols_; ++i2)line_index_[i2] += 1;
            return;
        }
        for (unsigned int i1 = line_index_[j]; i1 < line_index_[j + 1]; ++i1)   //if it already exists, just modify the value. 
        {
            if (elements_[i1].row() == i)
            {
                elements_[i1].setValue(value);
                return;
            }
        }
        elements_.insert(elements_.begin() + line_index_[j], Trituple<Scalar>(i, j, value));    //if it not exists, just insert the first place of the col
        for (unsigned int i2 = j + 1; i2 <= cols_; ++i2)line_index_[i2] += 1;
        return ;
    }
    else{
        if (line_index_[i] == line_index_[i + 1])          //the same with above
        {
            unsigned int i1 = line_index_[i];
            elements_.insert(elements_.begin() + i1, Trituple<Scalar>(i, j, value));
            for (unsigned int i2 = i + 1; i2 <= rows_; ++i2)line_index_[i2] += 1;
            return;
        }
        for (unsigned int i1 = line_index_[i]; i1 < line_index_[i + 1]; ++i1)
        {
            if (elements_[i1].col() == j)
            {
                elements_[i1].setValue(value);
                return;
            }
        }
        elements_.insert(elements_.begin() + line_index_[i], Trituple<Scalar>(i, j, value));
        for (unsigned int i2 = i + 1; i2 <= rows_; ++i2)line_index_[i2] += 1;
        return ;
    }
    
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    (*ptr_eigen_sparse_matrix_).coeffRef(i,j) = value;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator+ (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    SparseMatrix<Scalar> result(mat2);
    for(unsigned int i=0; i<elements_.size(); ++i)
    {
        unsigned int x = elements_[i].row();
        unsigned int y = elements_[i].col();
        Scalar v = elements_[i].value();
        v += result(x,y);
        result.setEntry(x,y,v);
    }
    return result;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    PHYSIKA_ASSERT(mat2.rows()==this->rows() && mat2.cols()==this->cols());
    SparseMatrix<Scalar> result(this->rows(), this->cols());
    (*result.ptr_eigen_sparse_matrix_) =  (*ptr_eigen_sparse_matrix_) + (*(mat2.ptr_eigen_sparse_matrix_));
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator+= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    for(unsigned int i=0; i<mat2.elements_.size(); ++i)
    {
        unsigned int x = mat2.elements_[i].row();
        unsigned int y = mat2.elements_[i].col();
        Scalar v = mat2.elements_[i].value();
        v += this->operator()(x, y);
        this->setEntry(x, y, v);
    }
    return *this;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    PHYSIKA_ASSERT(mat2.rows()==this->rows() && mat2.cols()==this->cols());
    (*ptr_eigen_sparse_matrix_) += (*(mat2.ptr_eigen_sparse_matrix_));
    return *this;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator- (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    SparseMatrix<Scalar> result(*this);
    for (unsigned int i = 0; i<mat2.elements_.size(); ++i)
    {
        unsigned int x = mat2.elements_[i].row();
        unsigned int y = mat2.elements_[i].col();
        Scalar v = mat2.elements_[i].value();
        v = result(x, y) - v;
        result.setEntry(x, y, v);
    }
    return result;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    PHYSIKA_ASSERT(mat2.rows()==this->rows() && mat2.cols()==this->cols());
    SparseMatrix<Scalar> result(this->rows(), this->cols());
    *(result.ptr_eigen_sparse_matrix_) = *(ptr_eigen_sparse_matrix_) - *(mat2.ptr_eigen_sparse_matrix_);
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator-= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    PHYSIKA_ASSERT(mat2.rows()==rows_ && mat2.cols()==cols_);
    for (unsigned int i = 0; i<mat2.elements_.size(); ++i)
    {
        unsigned int x = mat2.elements_[i].row();
        unsigned int y = mat2.elements_[i].col();
        Scalar v = mat2.elements_[i].value();
        v = (*this)(x, y) - v;
        this->setEntry(x, y, v);
    }
    return *this;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    PHYSIKA_ASSERT(mat2.rows()==this->rows() && mat2.cols()==this->cols());
    (*ptr_eigen_sparse_matrix_) -= (*(mat2.ptr_eigen_sparse_matrix_));
    return *this;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator= (const SparseMatrix<Scalar> &mat2)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    rows_ = mat2.rows_;
    cols_ = mat2.cols_;
    elements_ = mat2.elements_;
    line_index_ = mat2.line_index_;
    if (priority_ != mat2.priority_)
    {
        if (priority_ == 0)
        {
            unsigned int *num = new unsigned int[mat2.rows_ + 10], *mark = new unsigned int[mat2.rows_ + 10];
            for (unsigned int i = 0; i < mat2.rows_ + 10; ++i){ num[i] = 0; mark[i] = 0; }
            for (unsigned int i = 0; i < mat2.elements_.size(); ++i)
            {
                num[elements_[i].row()] += 1;
            }
            line_index_[0] = mark[0] = 0;
            for (unsigned int i = 1; i <= mat2.rows_; ++i)
            {
                line_index_[i] = line_index_[i - 1] + num[i - 1];
                mark[i] = line_index_[i];
            }
            //reorder the elements vector
            for (unsigned int i = 0; i < mat2.elements_.size(); ++i)
            {
                //put an element into correspondent mark[mat2.elements.row()] position
                this->elements_[mark[mat2.elements_[i].row()]++] = mat2.elements_[i];
            }
            delete[] num;
            delete[] mark;
        }
        else{
            unsigned int *num = new unsigned int[mat2.rows_ + 10], *mark = new unsigned int[mat2.rows_ + 10];
            for (unsigned int i = 0; i < mat2.rows_ + 10; ++i){ num[i] = 0; mark[i] = 0; }
            for (unsigned int i = 0; i < mat2.elements_.size(); ++i)
            {
                num[elements_[i].col()] += 1;
            }
            line_index_[0] = mark[0] = 0;
            for (unsigned int i = 1; i <= mat2.cols_; ++i)
            {
                line_index_[i] = line_index_[i - 1] + num[i - 1];
                mark[i] = line_index_[i];
            }
            //reorder the elements vector
            for (unsigned int i = 0; i < mat2.elements_.size(); ++i)
            {
                //put an element into correspondent mark[mat2.elements.col()] position
                this->elements_[mark[mat2.elements_[i].col()]++] = mat2.elements_[i];
            }
            delete[] num;
            delete[] mark;
        }
    }
    return *this;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    (*ptr_eigen_sparse_matrix_) = (*(mat2.ptr_eigen_sparse_matrix_));
    return *this;
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::operator== (const SparseMatrix<Scalar> &mat2) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    if(rows_ != mat2.rows() || cols_ != mat2.cols())return false;
    if (this->nonZeros() != mat2.nonZeros())return false;
    for (unsigned int i=0; i < elements_.size(); ++i)
    {
        if(is_floating_point<Scalar>::value)
        {
            if(isEqual(elements_[i].value(),mat2(elements_[i].row(), elements_[i].col()))==false)
                return false;
        }
        else
        {
            if(elements_[i].value() != mat2(elements_[i].row(), elements_[i].col()))
                return false;
        }
    }
    return true;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    SparseMatrix<Scalar> result = *this - mat2;
    if(result.nonZeros() == 0)return true;
    else return false;
#endif
}

template <typename Scalar>
bool SparseMatrix<Scalar>::operator!= (const SparseMatrix<Scalar> &mat2) const
{
    if(*this == mat2)return false;
    else return true;
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator* (Scalar scale) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    SparseMatrix<Scalar> result(*this);
    for(unsigned int i=0;i<elements_.size();++i)
    {
        result.elements_[i].setValue(elements_[i].value()*scale);
    }
    return result;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    SparseMatrix<Scalar> result(this->rows(), this->cols());
    *(result.ptr_eigen_sparse_matrix_) = *(ptr_eigen_sparse_matrix_) * scale;
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator* (const SparseMatrix<Scalar> &mat2) const
{
    if(this->cols() != mat2.rows())
    {
        std::cerr<<"operator * between two SparseMatrixes failed because they don't match"<<std::endl;
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX 
    if (priority_ == ROW_MAJOR && mat2.priority_ == ROW_MAJOR)
    {
        SparseMatrix<Scalar> result(rows_,mat2.cols_,ROW_MAJOR);
        //Scalar *vector_row = new Scalar[mat2.cols_+1];
        for (unsigned int i = 0; i < rows_; ++i)
        {
            unsigned begin_left = line_index_[i], end_left = line_index_[i+1];        //postfix "left" means the multiplicand
            for (unsigned int j = begin_left; j < end_left; ++j)
            {
                unsigned int y_left = elements_[j].col();
                Scalar v_left = elements_[j].value();
                unsigned int begin_right = mat2.line_index_[y_left], end_right = mat2.line_index_[y_left + 1];
                for (unsigned k = begin_right; k < end_right; ++k)
                {
                    unsigned int result_y = mat2.elements_[k].col();                     //mat2(y_left, result_y) = v_right
                    Scalar v_right = mat2.elements_[k].value();                              //assume the average number of nonzeros in a row is SR
                    result.setEntry(i, result_y, result(i, result_y) + v_left*v_right);           //time complexity O(SR*SR*rows) = O(matrix_left_nonzeros * SR)
                }
            }		
        }
        return result;
    }
    else if (priority_ == ROW_MAJOR && mat2.priority_ == COL_MAJOR)
    {
        SparseMatrix<Scalar> mat_temp = mat2;
        return (*this)*mat_temp;
    }
    else if (priority_ == COL_MAJOR && mat2.priority_ == COL_MAJOR)
    {
        SparseMatrix<Scalar> result(rows_, mat2.cols_ ,COL_MAJOR);
        for (unsigned int i = 0; i < mat2.cols_; ++i)
        {
            unsigned int begin_right = mat2.line_index_[i], end_right = mat2.line_index_[i + 1];
            for (unsigned int j = begin_right; j < end_right; ++j)
            {
                unsigned x_right = mat2.elements_[j].row();
                Scalar v_right = mat2.elements_[j].value();
                unsigned int begin_left = line_index_[x_right], end_left = line_index_[x_right + 1];
                for (unsigned int k = begin_left; k < end_left; ++k)
                {
                    unsigned int result_x = elements_[k].row();
                    Scalar v_left = elements_[k].value();
                    result.setEntry(result_x, i, result(result_x, i) + v_left*v_right);
                }
            }
        }
        return result;
    }
    else if (priority_ == COL_MAJOR && mat2.priority_ == ROW_MAJOR)
    {
        SparseMatrix<Scalar> mat_temp(0,0,COL_MAJOR);
        return (*this)*mat_temp;
    }
    SparseMatrix<Scalar> default_return(0,0,ROW_MAJOR);
    return default_return;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    SparseMatrix<Scalar> result(this->rows(),mat2.cols());
    (*result.ptr_eigen_sparse_matrix_) = (*ptr_eigen_sparse_matrix_) * (*(mat2.ptr_eigen_sparse_matrix_));
    return result;
#endif
}

template <typename Scalar>
VectorND<Scalar> SparseMatrix<Scalar>::leftMultiVec (const VectorND<Scalar> &vec) const
{
    if (this->rows() != vec.dims())
    {
        std::cerr << "operator * between VectorND and SpaseMatrix failed because the two don't match" << std::endl;
        std::exit(EXIT_FAILURE);
    }
    VectorND<Scalar> result(this->cols(),0);
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for(unsigned int i=0;i<elements_.size();++i)
    {
        unsigned int x = elements_[i].row();
        unsigned int y = elements_[i].col();
        Scalar v = elements_[i].value();
        result[y] += v*vec[x];
    }
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    Scalar sum;
    for (unsigned int i = 0; i < this->cols(); ++i)
    {
        sum = 0;
        for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(*ptr_eigen_sparse_matrix_, i); it; ++it)
        {
            sum += it.value()*vec[it.row()];
        }
        result[i] = sum;
    }

#endif
    return result;
}

template <typename Scalar>
VectorND<Scalar> SparseMatrix<Scalar>::operator* (const VectorND<Scalar> &vec) const
{
    if(this->cols() != vec.dims())
    {
        std::cerr<<"operator * between SpaseMatrix and VectorND failed because the two don't match"<<std::endl;
        std::exit(EXIT_FAILURE);
    }
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    VectorND<Scalar> result(rows_, 0);
    for (unsigned int i = 0; i < elements_.size(); ++i)
    {
        unsigned int x=elements_[i].row();
        unsigned int y=elements_[i].col();
        Scalar v = elements_[i].value();
        result[x] += v*vec[y];
    }
    return result;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    VectorND<Scalar> result(this->rows(),0);
    for(unsigned int i=0;i<this->cols();++i)
    {
        for (typename Eigen::SparseMatrix<Scalar>::InnerIterator it(*ptr_eigen_sparse_matrix_, i); it; ++it)
        {
            Scalar value = it.value();
            unsigned int j = it.row();
            result[j] += value*vec[i];
        }
    }
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator*=(Scalar scale)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for (unsigned int i = 0; i<elements_.size(); ++i)
    {
        elements_[i].setValue(elements_[i].value()*scale);
    }
    return *this;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    *ptr_eigen_sparse_matrix_ = *ptr_eigen_sparse_matrix_ * scale;
    return *this;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar> SparseMatrix<Scalar>::operator/ (Scalar scale) const
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    SparseMatrix<Scalar> result(*this);
    for (unsigned int i = 0; i<elements_.size(); ++i)
    {
        result.elements_[i].setValue(elements_[i].value()/scale);
    }
    return result;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    SparseMatrix<Scalar> result(this->rows(),this->cols());
    *result.ptr_eigen_sparse_matrix_ = *ptr_eigen_sparse_matrix_ / scale;
    return result;
#endif
}

template <typename Scalar>
SparseMatrix<Scalar>& SparseMatrix<Scalar>::operator/=(Scalar scale)
{
#ifdef PHYSIKA_USE_BUILT_IN_SPARSE_MATRIX
    for (unsigned int i = 0; i<elements_.size(); ++i)
    {
        elements_[i].setValue(elements_[i].value()/scale);
    }
    return *this;
#elif defined(PHYSIKA_USE_EIGEN_SPARSE_MATRIX)
    *ptr_eigen_sparse_matrix_ = *ptr_eigen_sparse_matrix_ / scale;
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
