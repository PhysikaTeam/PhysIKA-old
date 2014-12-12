/*
 * @file matrixMxN_test.cpp 
 * @brief Test the MatrixMxN class.
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
#include "Physika_Core/Matrices/matrix_MxN.h"
#include "Physika_Core/Vectors/vector_Nd.h"
using namespace std;
using Physika::MatrixMxN;
using Physika::MatrixBase;
using Physika::VectorND;

int main()
{
  cout<<"MatrixMxN Test"<<endl;
  MatrixMxN<double> mat_3x2(3,2);
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j < 2; ++j)
      mat_3x2(i,j) = i*2+j;
  cout<<"A 3x2 matrix of double numbers:(mat1)"<<endl;
  cout<<mat_3x2;
  cout<<"Rows: "<<mat_3x2.rows()<<" Cols: "<<mat_3x2.cols()<<endl;
  MatrixBase *base_pointer = &mat_3x2;
  cout<<"Test polymorphism, rows from MatrixBase pointer: "<<base_pointer->rows()<<endl;
  cout<<"Test polymorphism, cols from MatrixBase pointer: "<<base_pointer->cols()<<endl;
  double entries[6];
  for(int i = 0; i < 6; ++i)
    entries[i] = 5-i;
  MatrixMxN<double> mat_2x3(2,3,entries);
  cout<<"A 2x3 matrix of double numbers:(mat2)"<<endl;
  cout<<mat_2x3;
  MatrixMxN<double> mat2_3x2(3,2,entries);
  cout<<"A 3x2 matrix of double numbers with the same data:(mat3)"<<endl;
  cout<<mat2_3x2;
  // cout<<"mat1 + mat2:"<<endl;
  // cout<<mat_3x2 + mat_2x3<<endl;
  cout<<"mat1 + mat3:"<<endl;
  cout<<mat_3x2 + mat2_3x2<<endl;
  cout<<"mat1 - mat3:"<<endl;
  cout<<mat_3x2 - mat2_3x2<<endl;
  if(mat_3x2 == mat2_3x2)
    cout<<"mat1 == mat3"<<endl;
  else
    cout<<"mat1 != mat3"<<endl;
  cout<<"mat1 * 2.0:"<<endl;
  cout<<mat_3x2*2.0<<endl;
  cout<<"mat2 / 1.5:"<<endl;
  cout<<mat_2x3/1.5<<endl;
  cout<<"mat1 transpose:"<<endl;
  cout<<mat_3x2.transpose()<<endl;
  // cout<<"mat1 inverse:"<<endl;
  // cout<<mat_3x2.inverse()<<endl;
  // cout<<"mat1 determinant:"<<endl;
  // cout<<mat_3x2.determinant()<<endl;
  MatrixMxN<double> mat_square(2,2,entries);
  cout<<"A 2x2 matrix of double numbers:(mat4)"<<endl;
  cout<<mat_square<<endl;
  cout<<"mat4 inverse:"<<endl;
  cout<<mat_square.inverse()<<endl;
  cout<<"mat4 determinant:"<<endl;
  cout<<mat_square.determinant()<<endl;
  cout<<"Copy constructor test: construct a new matrix from mat4"<<endl;
  cout<<MatrixMxN<double>(mat_square)<<endl;
  cout<<"mat1 resized to 2x3 matrix:"<<endl;
  mat_3x2.resize(2,3);
  for(int i = 0; i < 2; ++i)
    for(int j = 0; j <3; ++j)
      mat_3x2(i,j) = i*3+j;
  cout<<mat_3x2<<endl;
  cout<<"mat1 resized to 3x3 matrix:"<<endl;
  mat_3x2.resize(3,3);
  for(int i = 0; i < 3; ++i)
    for(int j = 0; j <3; ++j)
      mat_3x2(i,j) = i*3+j;
  cout<<mat_3x2<<endl;
  cout<<"SVD: [0.68,0.597;-0.211,0.823;0.566,-0.605]\n";
  mat_3x2.resize(3,2);
  mat_3x2(0,0) = 0.68; mat_3x2(0,1) = 0.597;
  mat_3x2(1,0) = -0.211; mat_3x2(1,1) = 0.823;
  mat_3x2(2,0) = 0.566; mat_3x2(2,1) = -0.605;
  VectorND<double> singular_values(2);
  MatrixMxN<double> left_singular_vectors(3,2),right_singular_vectors(2,2);
  mat_3x2.singularValueDecomposition(left_singular_vectors,singular_values,right_singular_vectors);
  cout<<"Singular values: "<<singular_values<<"\n";
  cout<<"Left singular vectors: "<<left_singular_vectors<<"\n";
  cout<<"Right singular vectors: "<<right_singular_vectors<<"\n";
  cout<<"Eigen decomposition: [0.68,-0.33,-0.27,-0.717,-0.687,0.0259;\n";
  cout<<" -0.211,0.536,0.0268,0.214,-0.198,0.678;\n";
  cout<<"0.566,-0.444,0.904,-0.967,-0.74,0.225;\n";
  cout<<"0.597,0.108,0.832,-0.514,-0.782,-0.408;\n";
  cout<<"0.823,-0.0452,0.271,-0.726,0.998,0.275;\n";
  cout<<"-0.605,0.258,0.435,0.608,-0.563,0.0486]\n";
  mat_3x2.resize(6,6);
  mat_3x2(0,0) = 0.68; mat_3x2(0,1) = -0.33; mat_3x2(0,2) = -0.27; mat_3x2(0,3) = -0.717; mat_3x2(0,4) = -0.687; mat_3x2(0,5) = 0.0259;
  mat_3x2(1,0) = -0.211; mat_3x2(1,1) = 0.536; mat_3x2(1,2) = 0.0268; mat_3x2(1,3) = 0.214; mat_3x2(1,4) = -0.198; mat_3x2(1,5) = 0.678;
  mat_3x2(2,0) = 0.566; mat_3x2(2,1) = -0.444; mat_3x2(2,2) = 0.904; mat_3x2(2,3) = -0.967; mat_3x2(2,4) = -0.74; mat_3x2(2,5) =  0.225;
  mat_3x2(3,0) = 0.597; mat_3x2(3,1) = 0.108; mat_3x2(3,2) = 0.832; mat_3x2(3,3) = -0.514; mat_3x2(3,4) = -0.782; mat_3x2(3,5) = -0.408;
  mat_3x2(4,0) = 0.823; mat_3x2(4,1) = -0.0452; mat_3x2(4,2) = 0.271; mat_3x2(4,3) = -0.726; mat_3x2(4,4) = 0.998; mat_3x2(4,5) =  0.275;
  mat_3x2(5,0) = -0.605; mat_3x2(5,1) = 0.258; mat_3x2(5,2) = 0.435; mat_3x2(5,3) = 0.608; mat_3x2(5,4) = -0.563; mat_3x2(5,5) = 0.0486;
  VectorND<double> eigen_values_real, eigen_values_imag;
  MatrixMxN<double> eigen_vectors_real, eigen_vectors_imag;
  mat_3x2.eigenDecomposition(eigen_values_real,eigen_values_imag,eigen_vectors_real,eigen_vectors_imag);
  cout<<"Real part of eigen vlaues are:\n";
  cout<<eigen_values_real<<"\n";
  cout<<"Imag part of eigen values are:\n";
  cout<<eigen_values_imag<<"\n";
  cout<<"Real part of eigen vectors are:\n";
  cout<<eigen_vectors_real<<"\n";
  cout<<"Imag part of eigen vectors are:\n";
  cout<<eigen_vectors_imag<<"\n";
  return 0;
}
