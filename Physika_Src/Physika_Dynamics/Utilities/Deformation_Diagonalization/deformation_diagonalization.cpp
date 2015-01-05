/*
 * @file deformation_diagonalization.cpp 
 * @brief diagonalization of the deformation gradient via a modified SVD,
 *        implementation of the technique proposed in the sca04 paper:
 *        <Invertible Finite Elements For Robust Simulation of Large Deformation>.
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

#include <limits>
#include <iostream>
#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Matrices/matrix_2x2.h"
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Core/Utilities/math_utilities.h"
#include "Physika_Dynamics/Utilities/Deformation_Diagonalization/deformation_diagonalization.h"

namespace Physika{

template <typename Scalar, int Dim>
DeformationDiagonalization<Scalar,Dim>::DeformationDiagonalization()
    :epsilon_(std::numeric_limits<Scalar>::epsilon())
{
}
 
template <typename Scalar, int Dim>
DeformationDiagonalization<Scalar,Dim>::DeformationDiagonalization(Scalar epsilon)
{
    setEpsilon(epsilon);
}

template <typename Scalar, int Dim>
DeformationDiagonalization<Scalar,Dim>::~DeformationDiagonalization()
{
}
 
template <typename Scalar, int Dim>
Scalar DeformationDiagonalization<Scalar,Dim>::epsilon() const
{
    return epsilon_;
}
  
template <typename Scalar, int Dim>
void DeformationDiagonalization<Scalar,Dim>::setEpsilon(Scalar epsilon)
{
    if(epsilon < 0)
    {
        std::cerr<<"Warning: invalid epsilon value, default value is used instead!\n";
        epsilon_ = std::numeric_limits<Scalar>::epsilon();
    }
    else
        epsilon_ = epsilon;
}
      
template <typename Scalar, int Dim>
void DeformationDiagonalization<Scalar,Dim>::diagonalizeDeformationGradient(const SquareMatrix<Scalar,Dim> &deform_grad, SquareMatrix<Scalar,Dim> &left_rotation,
                                                                            SquareMatrix<Scalar,Dim> &diag_deform_grad, SquareMatrix<Scalar,Dim> &right_rotation) const
{
    diagonalizationTrait(deform_grad,left_rotation,diag_deform_grad,right_rotation);
}
            
template <typename Scalar, int Dim>
void DeformationDiagonalization<Scalar,Dim>::diagonalizeDeformationGradient(const SquareMatrix<Scalar,Dim> &deform_grad, DiagonalizedDeformation &diagonalized_deformation) const
{
    diagonalizeDeformationGradient(deform_grad,diagonalized_deformation.left_rotation,diagonalized_deformation.diag_deform_grad,diagonalized_deformation.right_rotation);
}

template <typename Scalar, int Dim>
void DeformationDiagonalization<Scalar,Dim>::diagonalizationTrait(const SquareMatrix<Scalar,2> &deform_grad, SquareMatrix<Scalar,2> &left_rotation,
                                                                  SquareMatrix<Scalar,2> &diag_deform_grad, SquareMatrix<Scalar,2> &right_rotation) const
{
    //naming correspondence with that in paper: U, left_rotation; V, right_rotation
    SquareMatrix<Scalar,2> F_transpose_F = deform_grad.transpose()*deform_grad;
    //imag part is dummy because F^T*F is symmetric matrix
    Vector<Scalar,2> eigen_values_real, eigen_values_imag;
    SquareMatrix<Scalar,2> eigen_vectors_real, eigen_vectors_imag;
    F_transpose_F.eigenDecomposition(eigen_values_real,eigen_values_imag,eigen_vectors_real,eigen_vectors_imag);
    right_rotation = eigen_vectors_real; //V is right rotation
    //Case I, det(V) == -1: simply multiply a column of V by -1
    if(right_rotation.determinant() < 0)
    {
        //here we negate the first column
        right_rotation(0,0) *= -1;
        right_rotation(1,0) *= -1;
    }
    //diagonal entries
    diag_deform_grad(0,0) = eigen_values_real[0] > 0 ? sqrt(eigen_values_real[0]) : 0;
    diag_deform_grad(1,1) = eigen_values_real[1] > 0 ? sqrt(eigen_values_real[1]) : 0;
    diag_deform_grad(0,1) = diag_deform_grad(1,0) = 0;
    //std::cout<<diag_deform_grad<<"\n";
    //inverse of F^
    SquareMatrix<Scalar,2> diag_deform_grad_inverse(0);
    diag_deform_grad_inverse(0,0) = diag_deform_grad(0,0) > epsilon_ ? 1/diag_deform_grad(0,0) : 0;
    diag_deform_grad_inverse(1,1) = diag_deform_grad(1,1) > epsilon_ ? 1/diag_deform_grad(1,1) : 0;
    // U = F*V*inv(F^)
    left_rotation = deform_grad * right_rotation;
    left_rotation *= diag_deform_grad_inverse;
    //Case II, an entry of F^ is near zero
    //set the corresponding column of U to be orthonormal to other columns
    if(diag_deform_grad(0,0) < epsilon_ && diag_deform_grad(1,1) < epsilon_)
    {
        //extreme case: material has collapsed almost to a point
        //U = I
        left_rotation = SquareMatrix<Scalar,2>::identityMatrix();
    }
    else
    {
        bool done = false;
        for(unsigned int col = 0; col < 2; ++col)
        {
            unsigned int col_a = col, col_b = (col+1)%2;
            if(diag_deform_grad(col_a,col_a) < epsilon_)
            {
                //entry a of F^ is near zero, set column a of U to be orthonormal with  column b
                left_rotation(0,col_a) = left_rotation(1,col_b);
                left_rotation(1,col_a) = -left_rotation(0,col_b);
                //the orthonormal vector leads to |U|<0, need to negate it
                if(left_rotation.determinant() < 0)
                {
                    left_rotation(0,col_a) *= -1;
                    left_rotation(1,col_a) *= -1;
                }
                //std::cout<<col_a<<"near zero: "<<diag_deform_grad<<"\n";
                done = true;
                break;
            }
        }
        if(!done)
        {
            //Case III, det(U) = -1: negate the minimal element of F^ and corresponding column of U
            if(left_rotation.determinant() < 0)
            {
                unsigned int smallest_value_idx = (diag_deform_grad(0,0) < diag_deform_grad(1,1)) ? 0 : 1;
                //negate smallest singular value
                diag_deform_grad(smallest_value_idx,smallest_value_idx) *= -1;
                left_rotation(0,smallest_value_idx) *= -1;
                left_rotation(1,smallest_value_idx) *= -1;
                //std::cout<<"det(U)=-1 "<<smallest_value_idx<<" "<<diag_deform_grad<<"\n";
            }
        }
    }
}
    
template <typename Scalar, int Dim>
void DeformationDiagonalization<Scalar,Dim>::diagonalizationTrait(const SquareMatrix<Scalar,3> &deform_grad, SquareMatrix<Scalar,3> &left_rotation,
                                                                  SquareMatrix<Scalar,3> &diag_deform_grad, SquareMatrix<Scalar,3> &right_rotation) const
{
    //naming correspondence with that in paper: U, left_rotation; V, right_rotation
    SquareMatrix<Scalar,3> F_transpose_F = deform_grad.transpose()*deform_grad;
    //imag part is dummy because F^T*F is symmetric matrix
    Vector<Scalar,3> eigen_values_real, eigen_values_imag;
    SquareMatrix<Scalar,3> eigen_vectors_real, eigen_vectors_imag;
    F_transpose_F.eigenDecomposition(eigen_values_real,eigen_values_imag,eigen_vectors_real,eigen_vectors_imag);
    right_rotation = eigen_vectors_real; //V is right rotation
    //Case I, det(V) == -1: simply multiply a column of V by -1
    if(right_rotation.determinant() < 0)
    {
        //here we negate the first column
        for(unsigned int row = 0; row < 3; ++row)
            right_rotation(row,0) *= -1;
    }
    //diagonal entries
    for(unsigned int row = 0; row < 3; ++row)
        for(unsigned int col = 0; col < 3; ++col)
            if(row == col)
                diag_deform_grad(row,col) = eigen_values_real[row] > 0 ? sqrt(eigen_values_real[row]) : 0;
            else
                diag_deform_grad(row,col) = 0;
    //inverse of F^
    SquareMatrix<Scalar,3> diag_deform_grad_inverse(0);
    for(unsigned int row = 0; row < 3; ++row)
        diag_deform_grad_inverse(row,row) = diag_deform_grad(row,row) > epsilon_ ? 1/diag_deform_grad(row,row) : 0;
    // U = F*V*inv(F^)
    left_rotation = deform_grad * right_rotation;
    left_rotation *= diag_deform_grad_inverse;
    //Case II, an entry of F^ is near zero
    //set the corresponding column of U to be orthonormal to other columns
    bool extreme_case = true; // F = 0
    for(unsigned int row = 0; row < 3; ++row)
    {
        extreme_case =  extreme_case && (diag_deform_grad(row,row) < epsilon_);
        if(extreme_case == false)
            break;
    }
    if(extreme_case)
    {
        //extreme case: material has collapsed almost to a point
        //U = I
        left_rotation = SquareMatrix<Scalar,3>::identityMatrix();
    }
    else
    {
        bool done = false;
        for(unsigned int col = 0; col < 3; ++col)
        {
            unsigned int col_a = col, col_b = (col+1)%3, col_c = (col+2)%3;
            if(diag_deform_grad(col_b,col_b) < epsilon_ && diag_deform_grad(col_c,col_c) < epsilon_)
            {
                //two entries of F^ are near zero: set corresponding columns of U to be orthonormal to the remainning one
                Vector<Scalar,3> left_rotation_col_a;
                for(unsigned int row = 0; row < 3; ++row)
                    left_rotation_col_a[row] = left_rotation(row,col_a);
                Vector<Scalar,3> left_rotation_col_b; //b is chosen to be orthogonal to a
                unsigned int smallest_idx = 0;
                for(unsigned int row = 1; row < 3; ++row)
                    if(abs(left_rotation_col_a[row]) < abs(left_rotation_col_a[smallest_idx]))
                        smallest_idx = row;
                Vector<Scalar,3> axis(0);
                axis[smallest_idx] = 1;
                left_rotation_col_b = left_rotation_col_a.cross(axis);
                left_rotation_col_b.normalize();
                Vector<Scalar,3> left_rotation_col_c = left_rotation_col_a.cross(left_rotation_col_b);
                left_rotation_col_c.normalize();
                for(unsigned int row = 0; row < 3; ++row)
                {
                    left_rotation(row,col_b) = left_rotation_col_b[row];
                    left_rotation(row,col_c) = left_rotation_col_c[row];
                }
                //the orthonormal vector leads to |U|<0, need to negate it
                if(left_rotation.determinant() < 0)
                {
                    for(unsigned int row = 0; row < 3; ++row)
                        left_rotation(row,col_b) *= -1;
                }
                done = true;
                break;
            }
        }
        if(!done)
        {
            for(unsigned int col = 0; col < 3; ++col)
            {
                unsigned int col_a = col, col_b = (col+1)%3, col_c = (col+2)%3;
                if(diag_deform_grad(col_a,col_a) < epsilon_)
                {
                    //only one entry of F^ is near zero
                    Vector<Scalar,3> left_rotation_col_b, left_rotation_col_c;
                    for(unsigned int row = 0; row < 3; ++row)
                    {
                        left_rotation_col_b[row] = left_rotation(row,col_b);
                        left_rotation_col_c[row] = left_rotation(row,col_c);
                    }
                    Vector<Scalar,3> left_rotation_col_a = left_rotation_col_b.cross(left_rotation_col_c);
                    left_rotation_col_a.normalize();
                    for(unsigned int row = 0; row < 3; ++row)
                        left_rotation(row,col_a) = left_rotation_col_a[row];
                    //the orthonormal vector leads to |U|<0, need to negate it
                    if(left_rotation.determinant() < 0)
                    {
                        for(unsigned int row = 0; row < 3; ++row)
                            left_rotation(row,col_a) *= -1;
                    }
                    done = true;
                    break;
                }
            }
        }
        if(!done)
        {
            //Case III, det(U) = -1: negate the minimal element of F^ and corresponding column of U
            if(left_rotation.determinant() < 0)
            {
                unsigned int smallest_value_idx = 0;
                for(unsigned int i = 1; i < 3; ++i)
                    if(diag_deform_grad(i,i) < diag_deform_grad(smallest_value_idx,smallest_value_idx))
                        smallest_value_idx = i;
                //negate smallest singular value
                diag_deform_grad(smallest_value_idx,smallest_value_idx) *= -1;
                for(unsigned int i = 0; i < 3; ++i)
                    left_rotation(i,smallest_value_idx) *= -1;
            }
        }
    }
}

//explicit instantiations
template class DeformationDiagonalization<float,2>;
template class DeformationDiagonalization<float,3>;
template class DeformationDiagonalization<double,2>;
template class DeformationDiagonalization<double,3>;
    
}  //end of namespace Physika
