/*
 * @file matrix2x2_test.cpp 
 * @brief Test the Matrix2x2 class.
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
#include "Physika_Core/Matrices/matrix_2x2.h"
using namespace std;
using Physika::Matrix2x2;
using Physika::MatrixBase;

int main()
{
  cout<<"Matrix2x2 Test"<<endl;
  Matrix2x2<double> mat_double(2.0,1.0,1.0,2.0);
  cout<<"A 2x2 matrix of double numbers:"<<endl;
  cout<<mat_double;
  cout<<"Rows: "<<mat_double.rows()<<" Cols: "<<mat_double.cols()<<endl;
  MatrixBase *base_pointer = &mat_double;
  cout<<"Test polymorphism, rows from MatrixBase pointer: "<<base_pointer->rows()<<endl;
  cout<<"Test polymorphism, cols from MatrixBase pointer: "<<base_pointer->cols()<<endl;
  Matrix2x2<double> mat_double2(0.0,1.0,3.0,0.0);
  cout<<"Another 2x2 matrix of double numbers:"<<endl;
  cout<<mat_double2;
  cout<<"matrix1 add matrix2: (+)"<<endl;
  cout<<mat_double+mat_double2<<endl;
  cout<<"matrix1 add matrix2: (+=)"<<endl;
  cout<<(mat_double+=mat_double2)<<endl;
  cout<<"matrix1 sub matrix2: (-)"<<endl;
  cout<<mat_double-mat_double2<<endl;
  cout<<"matrix1 sub matrix2: (-=)"<<endl;
  cout<<(mat_double-=mat_double2)<<endl;
  cout<<"equal test: (matrix1 == matrix2)?"<<endl;
  if(mat_double==mat_double2)
      cout<<"Yes"<<endl;
  else
      cout<<"No"<<endl;
  cout<<"matrix1 copy to matrix2:"<<endl;
  mat_double2=mat_double;
  cout<<mat_double2<<endl;
  cout<<"matrix1 mult scalar(2): (*)"<<endl;
  cout<<mat_double*2.0f<<endl;
  cout<<"scalar(2) mult matrix: (*)"<<endl;
  cout<<2.0f*mat_double<<endl;
  cout<<"matrix1 mult scalar(2): (*=)"<<endl;
  cout<<(mat_double*=2.0f)<<endl;
  cout<<"matrix1 div scalar(4): (/)"<<endl;
  cout<<mat_double/4.0f<<endl;
  cout<<"matrix1 div scalar(4): (/=)"<<endl;
  cout<<(mat_double/=4.0f)<<endl;
  cout<<"matrix1 transpose:"<<endl;
  cout<<mat_double.transpose()<<endl;
  cout<<"matrix1 inverse:"<<endl;
  cout<<mat_double.inverse()<<endl;
  cout<<"matrix1 determinant:"<<endl;
  cout<<mat_double.determinant()<<endl;

  return 0;
}
