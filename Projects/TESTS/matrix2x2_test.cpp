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

int main()
{
  cout<<"Matrix2x2 Test"<<endl;
  Matrix2x2<float> mat_float(2.0f,1.0f,1.0f,2.0f);
  cout<<"A 2x2 matrix of float numbers:"<<endl;
  cout<<mat_float;
  cout<<"Rows: "<<mat_float.rows()<<" Cols: "<<mat_float.cols()<<endl;
  Matrix2x2<float> mat_float2(0.0f,1.0f,1.0f,0.0f);
  cout<<"Another 2x2 matrix of float numbers:"<<endl;
  cout<<mat_float2;
  cout<<"matrix1 add matrix2:"<<endl;
  cout<<mat_float+mat_float2<<endl;
  cout<<"matrix1 sub matrix2:"<<endl;
  cout<<mat_float-mat_float2<<endl;
  cout<<"matrix1 copy to matrix2:"<<endl;
  mat_float2=mat_float;
  cout<<mat_float2<<endl;
  cout<<"matrix1 mult scalar(2):"<<endl;
  cout<<mat_float*2.0f<<endl;
  cout<<"matrix1 div scalar(4):"<<endl;
  cout<<mat_float/4.0f<<endl;
  cout<<"matrix1 transpose:"<<endl;
  cout<<mat_float.transpose()<<endl;
  cout<<"matrix1 inverse:"<<endl;
  cout<<mat_float.inverse()<<endl;
  return 0;
}
