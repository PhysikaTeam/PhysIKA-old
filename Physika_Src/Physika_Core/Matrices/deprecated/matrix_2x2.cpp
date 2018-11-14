/*
 * @file matrix_2x2.cpp
 * @brief 2x2 matrix.
 * @author Fei Zhu, Wei Chen
 *
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0.
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <Physika_Dependency/Eigen/Eigen>

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"

namespace Physika{
    
template <typename Scalar>
void SquareMatrix<Scalar,2>::singularValueDecomposition(SquareMatrix<Scalar,2> &left_singular_vectors,
                                                        Vector<Scalar,2> &singular_values,
                                                        SquareMatrix<Scalar,2> &right_singular_vectors) const
{
    //hack: Eigen::SVD does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,2,2> temp_matrix;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            temp_matrix(i,j) = static_cast<long double>((*this)(i,j));
    Eigen::JacobiSVD<Eigen::Matrix<long double,2,2> > svd(temp_matrix,Eigen::ComputeFullU|Eigen::ComputeFullV);
    const Eigen::Matrix<long double,2,2> &left = svd.matrixU(), &right = svd.matrixV();
    const Eigen::Matrix<long double,2,1> &values = svd.singularValues();
    for(unsigned int i = 0; i < 2; ++i)
    {
        singular_values[i] = static_cast<Scalar>(values(i,0));
        for(unsigned int j = 0; j < 2; ++j)
        {
            left_singular_vectors(i,j) = static_cast<Scalar>(left(i,j));
            right_singular_vectors(i,j) = static_cast<Scalar>(right(i,j));
        }
    }
}

template <typename Scalar>
void SquareMatrix<Scalar,2>::singularValueDecomposition(SquareMatrix<Scalar,2> &left_singular_vectors,
                                                        SquareMatrix<Scalar,2> &singular_values_diagonal,
                                                        SquareMatrix<Scalar,2> &right_singular_vectors) const
{
    Vector<Scalar,2> singular_values;
    singularValueDecomposition(left_singular_vectors,singular_values,right_singular_vectors);
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            singular_values_diagonal(i,j) = (i==j) ? singular_values[i] : 0;
}

template <typename Scalar>
void SquareMatrix<Scalar,2>::eigenDecomposition(Vector<Scalar,2> &eigen_values_real, 
                                                Vector<Scalar,2> &eigen_values_imag,
                                                SquareMatrix<Scalar,2> &eigen_vectors_real, 
                                                SquareMatrix<Scalar,2> &eigen_vectors_imag)
{
    //hack: Eigen::EigenSolver does not support integer types, hence we cast Scalar to long double for decomposition
    Eigen::Matrix<long double,2,2> temp_matrix;
    for(unsigned int i = 0; i < 2; ++i)
        for(unsigned int j = 0; j < 2; ++j)
            temp_matrix(i,j) = static_cast<long double>((*this)(i,j));
    Eigen::EigenSolver<Eigen::Matrix<long double,2,2> > eigen(temp_matrix);
    Eigen::Matrix<std::complex<long double>,2,2> vectors = eigen.eigenvectors();
    const Eigen::Matrix<std::complex<long double>,2,1> &values = eigen.eigenvalues();
    for(unsigned int i = 0; i < 2; ++i)
    {
        eigen_values_real[i] = static_cast<Scalar>(values(i,0).real());
        eigen_values_imag[i] = static_cast<Scalar>(values(i,0).imag());
        for(unsigned int j = 0; j < 2; ++j)
        {
            eigen_vectors_real(i,j) = static_cast<Scalar>(vectors(i,j).real());
            eigen_vectors_imag(i,j) = static_cast<Scalar>(vectors(i,j).imag());
        }
    }
}

//explicit instantiation of template so that it could be compiled into a lib
template class SquareMatrix<unsigned char, 2>;
template class SquareMatrix<unsigned short, 2>;
template class SquareMatrix<unsigned int, 2>;
template class SquareMatrix<unsigned long, 2>;
template class SquareMatrix<unsigned long long, 2>;
template class SquareMatrix<signed char, 2>;
template class SquareMatrix<short, 2>;
template class SquareMatrix<int, 2>;
template class SquareMatrix<long, 2>;
template class SquareMatrix<long long, 2>;
template class SquareMatrix<float, 2>;
template class SquareMatrix<double, 2>;
template class SquareMatrix<long double, 2>;

}//end of namespace Physika