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
using namespace std;
using Physika::MatrixMxN;
using Physika::MatrixBase;

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
  cout<<"mat1 + mat2:"<<endl;
  cout<<mat_3x2 + mat_2x3<<endl;
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
  cout<<"mat1 inverse:"<<endl;
  cout<<mat_3x2.inverse()<<endl;
  cout<<"mat1 determinant:"<<endl;
  cout<<mat_3x2.determinant()<<endl;
  MatrixMxN<double> mat_square(2,2,entries);
  cout<<"A 2x2 matrix of double numbers:(mat4)"<<endl;
  cout<<mat_square<<endl;
  cout<<"mat4 inverse:"<<endl;
  cout<<mat_square.inverse()<<endl;
  cout<<"mat4 determinant:"<<endl;
  cout<<mat_square.determinant()<<endl;
  cout<<"Copy constructor test: construct a new matrix from mat4"<<endl;
  cout<<MatrixMxN<double>(mat_square)<<endl;
  return 0;
}
