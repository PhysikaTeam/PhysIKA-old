/*
 * @file matrix4x4_test.cpp 
 * @brief Test the Matrix4x4 class.
 * @author Sheng Yang, Liyou Xu
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
#include "Physika_Core/Matrices/matrix_4x4.h"
using namespace std;

int main()
{
  cout<<"Matrix4x4 Test"<<endl;
  Physika::SquareMatrix<double,4> mat_double(3.0,1.0,1.0,1.0, 3.0,3.0,1.0,1.0, 3.0,3.0,3.0,1.0, 3.0,3.0,3.0,3.0);
  cout<<"A 4x4 matrix of double numbers:"<<endl;
  cout<<mat_double;
  cout<<"Rows: "<<mat_double.rows()<<" Cols: "<<mat_double.cols()<<endl;
  Physika::SquareMatrix<double,4> mat_double3(1.0,1.0,1.0,1.0, 0.0,3.0,3.0,0.0, 3.0,2.0,3.0,1.0, 1.0,1.0,0.0,0.0);
  cout<<"Another 4x4 matrix of double numbers:"<<endl;
  cout<<mat_double3<<endl;
  cout<<"matrix2[1][1]:"<<mat_double3(1,1)<<endl;
  cout<<"matrix2's trace:"<<mat_double3.trace()<<endl;
  cout<<"matrix1 and matrix2's double contraction:"<<mat_double.doubleContraction(mat_double3)<<endl;
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
  cout<<"copy matrix2 to matrix1:"<<endl;
  mat_double = mat_double3;
  cout<<mat_double<<endl;
  cout<<"matrix1 inverse:"<<endl;
  cout<<mat_double.inverse()<<endl;
  cout<<"matrix1 determinant:"<<endl;
  cout<<mat_double.determinant()<<endl;
  cout<<"matrix1^(-1) mult matrix2:"<<endl;
  cout<<mat_double.inverse()*mat_double3<<endl;
  int a;
  cin>>a;
  return 0;
}
