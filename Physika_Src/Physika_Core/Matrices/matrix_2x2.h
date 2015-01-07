/*
 * @file matrix_2x2.h 
 * @brief 2x2 matrix.
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

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * SquareMatrix<Scalar,2> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar,2>: public MatrixBase
{
public:
    SquareMatrix();
    explicit SquareMatrix(Scalar);
    SquareMatrix(Scalar x00, Scalar x01, Scalar x10, Scalar x11);
    SquareMatrix(const Vector<Scalar,2> &row1, const Vector<Scalar,2> &row2);
    SquareMatrix(const SquareMatrix<Scalar,2> &);
    ~SquareMatrix();
    inline unsigned int rows() const{return 2;}
    inline unsigned int cols() const{return 2;}
    Scalar& operator() (unsigned int i, unsigned int j);
    const Scalar& operator() (unsigned int i, unsigned int j) const;
    SquareMatrix<Scalar,2> operator+ (const SquareMatrix<Scalar,2> &) const;
    SquareMatrix<Scalar,2>& operator+= (const SquareMatrix<Scalar,2> &);
    SquareMatrix<Scalar,2> operator- (const SquareMatrix<Scalar,2> &) const;
    SquareMatrix<Scalar,2>& operator-= (const SquareMatrix<Scalar,2> &);
    SquareMatrix<Scalar,2>& operator= (const SquareMatrix<Scalar,2> &);
    bool operator== (const SquareMatrix<Scalar,2> &) const;
    bool operator!= (const SquareMatrix<Scalar,2> &) const;
    SquareMatrix<Scalar,2> operator* (Scalar) const;
    SquareMatrix<Scalar,2>& operator*= (Scalar);
    Vector<Scalar,2> operator* (const Vector<Scalar,2> &) const;
    SquareMatrix<Scalar,2> operator* (const SquareMatrix<Scalar,2> &) const;
    SquareMatrix<Scalar,2>& operator*= (const SquareMatrix<Scalar,2> &);
    SquareMatrix<Scalar,2> operator/ (Scalar) const;
    SquareMatrix<Scalar,2>& operator/= (Scalar);
    SquareMatrix<Scalar,2> transpose() const;
    SquareMatrix<Scalar,2> inverse() const;
    Scalar determinant() const;
    Scalar trace() const;
    Scalar doubleContraction(const SquareMatrix<Scalar,2> &) const;//double contraction
    void singularValueDecomposition(SquareMatrix<Scalar,2> &left_singular_vectors,
                                    Vector<Scalar,2> &singular_values,   //singluar values are in descending order
                                    SquareMatrix<Scalar,2> &right_singular_vectors) const;
    void eigenDecomposition(Vector<Scalar,2> &eigen_values_real, Vector<Scalar,2> &eigen_values_imag,
                            SquareMatrix<Scalar,2> &eigen_vectors_real, SquareMatrix<Scalar,2> &eigen_vectors_imag);

    static SquareMatrix<Scalar,2> identityMatrix();
 
protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,2,2,Eigen::DontAlign> eigen_matrix_2x2_;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar data_[2][2];
#endif
private:
    void compileTimeCheck()
    {
        //SquareMatrix<Scalar,Dim> is only defined for element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "SquareMatrix<Scalar,2> are only defined for integer types and floating-point types.");
    }

};

//overriding << for SquareMatrix<Scalar,2>
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,2> &mat)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
    {
        s<<"["<<static_cast<int>(mat(0,0))<<", "<<static_cast<int>(mat(0,1))<<"; ";
        s<<static_cast<int>(mat(1,0))<<", "<<static_cast<int>(mat(1,1))<<"]";
    }
    else
    {
        s<<"["<<mat(0,0)<<", "<<mat(0,1)<<"; ";
        s<<mat(1,0)<<", "<<mat(1,1)<<"]";
    }
    return s;
}

//make * operator commutative
template <typename S, typename T>
inline SquareMatrix<T,2> operator* (S scale, const SquareMatrix<T,2> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef SquareMatrix<float,2> Matrix2f;
typedef SquareMatrix<double,2> Matrix2d;
typedef SquareMatrix<int,2> Matrix2i;

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_2X2_H_
