/*
 * @file sparse_matrix_internal.h 
 * @brief internal data structure used by SparseMatrix.
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

#ifndef PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_INTERNAL_H_
#define PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_INTERNAL_H_

namespace Physika{

namespace SparseMatrixInternal{

//storage mode of sparse matrix    
enum SparseMatrixStoreMode{
    ROW_MAJOR,
    COL_MAJOR
};

/*
 * class Trituple is used to store a node's information in the orthogonal list
 */
template <typename Scalar>
class Trituple
{
public:
    Trituple():row_(0),col_(0),value_(0),row_next_(NULL),col_next_(NULL){}
    Trituple(unsigned int row, unsigned int col, Scalar value)
        :row_(row),col_(col),value_(value),row_next_(NULL),col_next_(NULL){}
    bool operator==(const Trituple<Scalar> &tri2) const
    {
        if(tri2.row_ == row_ && tri2.col_ == col_ && tri2.value_ == value_)
            return true;
        return false;	
    }
    bool operator!=(const Trituple<Scalar> &tri2) const
    {
        if(tri2.row_ != row_ || tri2.col_ != col_ || tri2.value_ != value_)
            return true;
        return false;		
    }
    inline unsigned int row() const { return row_;}
    inline unsigned int col() const { return col_;}
    inline Scalar value() const { return value_;}
    inline void setRow(unsigned int i){row_ = i;}
    inline void setCol(unsigned int j){col_ = j;}
    inline void setValue(Scalar k){value_ = k;}
private:
    unsigned int row_;
    unsigned int col_;
    Scalar value_;
    Trituple<Scalar> *row_next_;
    Trituple<Scalar> *col_next_;
};

//overridding << for SparseMatrixInternal::Trituple<Scalar>
template <typename Scalar>
std::ostream& operator<<(std::ostream &s, const SparseMatrixInternal::Trituple<Scalar> &tri)
{
    s<<"<"<<tri.row()<<", "<<tri.col()<<", "<<tri.value()<<">";
    return s;
}

}  //end of namespace SparseMatrixInternal

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_SPARSE_MATRIX_INTERNAL_H_
