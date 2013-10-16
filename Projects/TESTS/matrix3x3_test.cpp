/*
 * @file matrix3x3_test.cpp 
 * @brief Test the Matrix3x3 class.
 * @author Sheng Yang
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
#include "Physika_Core/Matrices/matrix_3x3.h"
using namespace std;
using Physika::Matrix3x3;

int main()
{
  cout<<"Matrix3x3 Test"<<endl;
  Matrix3x3<double> mat_double(3.0, 1.0,1.0,3.0,3,0,1.0,3.0,3.0);
  cout<<"A 3x3 matrix of double numbers:"<<endl;
  cout<<mat_double;
  cout<<"Rows: "<<mat_double.rows()<<" Cols: "<<mat_double.cols()<<endl;
  Matrix3x3<double> mat_double3(0.0,1.0,1.0,0.0,3.0,3.0,3.0,2.0,3.0);
  cout<<"Another 3x3 matrix of double numbers:"<<endl;
  cout<<mat_double3;
  cout<<"matrix1 add matrix3: (+)"<<endl;
  cout<<mat_double+mat_double3<<endl;
  cout<<"matrix1 add matrix3: (+=)"<<endl;
  cout<<(mat_double+=mat_double3)<<endl;
  cout<<"matrix1 sub matrix3: (-)"<<endl;
  cout<<mat_double-mat_double3<<endl;
  cout<<"matrix1 sub matrix3: (-=)"<<endl;
  cout<<(mat_double-=mat_double3)<<endl;
  cout<<"equal test: (matrix1 == matrix3)?"<<endl;
  if(mat_double==mat_double3)
      cout<<"Yes"<<endl;
  else
      cout<<"No"<<endl;
  cout<<"matrix1 copy to matrix3:"<<endl;
  mat_double3=mat_double;
  cout<<mat_double3<<endl;
  cout<<"matrix1 mult scalar(3): (*)"<<endl;
  cout<<mat_double*3.0f<<endl;
  cout<<"matrix1 mult scalar(3): (*=)"<<endl;
  cout<<(mat_double*=3.0f)<<endl;
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
  int a;
  cin>>a;
  return 0;
}
