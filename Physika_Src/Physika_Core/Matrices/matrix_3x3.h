/*
 * @file matrix_3x3.h
 * @brief 3x3 matrix.
 * @author Sheng Yang, Fei Zhu
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_
#define PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_

#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/type_utilities.h"
#include "Physika_Core/Matrices/square_matrix.h"

namespace Physika{

template <typename Scalar, int Dim> class Vector;

/*
 * SquareMatrix<Scalar,3> are defined for C++ fundamental integer types and floating-point types
 */

template <typename Scalar>
class SquareMatrix<Scalar,3>: public MatrixBase
{
public:
    SquareMatrix();
    explicit SquareMatrix(Scalar);
    SquareMatrix(Scalar x00, Scalar x01, Scalar x02, Scalar x10, Scalar x11, Scalar x12, Scalar x20, Scalar x21, Scalar x22);
    SquareMatrix(const Vector<Scalar,3> &row1, const Vector<Scalar,3> &row2, const Vector<Scalar,3> &row3);
    SquareMatrix(const SquareMatrix<Scalar,3>&);
    ~SquareMatrix();
    inline unsigned int rows() const{return 3;}
    inline unsigned int cols() const{return 3;}
    Scalar& operator() (unsigned int i, unsigned int j );
    const Scalar& operator() (unsigned int i, unsigned int j) const;
    Vector<Scalar,3> rowVector(unsigned int i) const;
    Vector<Scalar,3> colVector(unsigned int i) const;
    SquareMatrix<Scalar,3> operator+ (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3>& operator+= (const SquareMatrix<Scalar,3> &);
    SquareMatrix<Scalar,3> operator- (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3>& operator-= (const SquareMatrix<Scalar,3> &);
    SquareMatrix<Scalar,3>& operator= (const SquareMatrix<Scalar,3> &);
    bool operator== (const SquareMatrix<Scalar,3> &) const;
    bool operator!= (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3> operator* (Scalar) const;
    SquareMatrix<Scalar,3>& operator*= (Scalar);
    Vector<Scalar,3> operator* (const Vector<Scalar,3> &) const;
    SquareMatrix<Scalar,3> operator* (const SquareMatrix<Scalar,3> &) const;
    SquareMatrix<Scalar,3>& operator*= (const SquareMatrix<Scalar,3> &);
    SquareMatrix<Scalar,3> operator/ (Scalar) const;
    SquareMatrix<Scalar,3>& operator/= (Scalar);
    SquareMatrix<Scalar,3> transpose() const;
    SquareMatrix<Scalar,3> inverse() const;
    Scalar determinant() const;
    Scalar trace() const;
    Scalar doubleContraction(const SquareMatrix<Scalar,3> &) const;//double contraction
    Scalar frobeniusNorm() const;
    void singularValueDecomposition(SquareMatrix<Scalar,3> &left_singular_vectors,
                                    Vector<Scalar,3> &singular_values,   //singluar values are in descending order
                                    SquareMatrix<Scalar,3> &right_singular_vectors) const;
    void singularValueDecomposition(SquareMatrix<Scalar,3> &left_singular_vectors,
                                    SquareMatrix<Scalar,3> &singular_values_diagonal,   //singluar values in descending order as a diagonal matrix
                                    SquareMatrix<Scalar,3> &right_singular_vectors) const;
    void eigenDecomposition(Vector<Scalar,3> &eigen_values_real, Vector<Scalar,3> &eigen_values_imag,
                            SquareMatrix<Scalar,3> &eigen_vectors_real, SquareMatrix<Scalar,3> &eigen_vectors_imag);

    static SquareMatrix<Scalar,3> identityMatrix();

protected:
#ifdef PHYSIKA_USE_EIGEN_MATRIX
    Eigen::Matrix<Scalar,3,3> eigen_matrix_3x3_;
#elif defined(PHYSIKA_USE_BUILT_IN_MATRIX)
    Scalar data_[3][3];
#endif
private:
    void compileTimeCheck()
    {
        //SquareMatrix<Scalar,Dim> is only defined for element type of integers and floating-point types
        //compile time check
        PHYSIKA_STATIC_ASSERT((is_integer<Scalar>::value||is_floating_point<Scalar>::value),
                              "SquareMatrix<Scalar,3> are only defined for integer types and floating-point types.");
    }

};

//overriding << for SquareMatrix<Scalar,3>
template <typename Scalar>
inline std::ostream& operator<< (std::ostream &s, const SquareMatrix<Scalar,3> &mat)
{
    if((is_same<Scalar,unsigned char>::value)||(is_same<Scalar,signed char>::value))
    {
        s<<"["<<static_cast<int>(mat(0,0))<<", "<<static_cast<int>(mat(0,1))<<", "<<static_cast<int>(mat(0,2))<<"; ";
        s<<static_cast<int>(mat(1,0))<<", "<<static_cast<int>(mat(1,1))<<", "<<static_cast<int>(mat(1,2))<<"; ";
        s<<static_cast<int>(mat(2,0))<<", "<<static_cast<int>(mat(2,1))<<", "<<static_cast<int>(mat(2,2))<<"]";
    }
    else
    {
        s<<"["<<mat(0,0)<<", "<<mat(0,1)<<", "<<mat(0,2)<<"; ";
        s<<mat(1,0)<<", "<<mat(1,1)<<", "<<mat(1,2)<<"; ";
        s<<mat(2,0)<<", "<<mat(2,1)<<", "<<mat(2,2)<<"]";
    }
    return s;
}

//make * operator commutative
template <typename S, typename T>
inline SquareMatrix<T,3> operator* (S scale, const SquareMatrix<T,3> &mat)
{
    return mat*scale;
}

//convenient typedefs
typedef SquareMatrix<float,3> Matrix3f;
typedef SquareMatrix<double,3> Matrix3d;
typedef SquareMatrix<int,3> Matrix3i;

}  //end of namespace Physika

#endif //PHYSIKA_CORE_MATRICES_MATRIX_3X3_H_
