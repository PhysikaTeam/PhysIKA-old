/*
 * @file vector2d_test.cpp 
 * @brief Test the Vector2D class.
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
#include "Physika_Core/Vectors/vector_2d.h"
//#include "Physika_Core/Vectors/vector.h"
using namespace std;

int main()
{
  cout<<"Vector2D Test"<<endl;
  Physika::Vector2D(double) vec_double(2.0,1.0);
  cout<<"A 2d vector of double numbers:"<<endl;
  cout<<vec_double;
  cout<<"Dims: "<<vec_double.dims()<<endl;
  Physika::Vector2D(double) vec_double2(0.0,1.0);
  cout<<"Another 2d vector of double numbers:"<<endl;
  cout<<vec_double2;
  cout<<"vector1 add vector2: (+)"<<endl;
  cout<<vec_double+vec_double2<<endl;
  cout<<"vector1 add vector2: (+=)"<<endl;
  cout<<(vec_double+=vec_double2)<<endl;
  cout<<"vector1 sub vector2: (-)"<<endl;
  cout<<vec_double-vec_double2<<endl;
  cout<<"vector1 sub vector2: (-=)"<<endl;
  cout<<(vec_double-=vec_double2)<<endl;
  cout<<"equal test: (vector1 == vector2)?"<<endl;
  if(vec_double==vec_double2)
      cout<<"Yes"<<endl;
  else
      cout<<"No"<<endl;
  cout<<"vector1 copy to vector2:"<<endl;
  vec_double2=vec_double;
  cout<<vec_double2<<endl;
  cout<<"vector1 mult scalar(2): (*)"<<endl;
  cout<<vec_double*2.0<<endl;
  cout<<"vector1 mult scalar(2): (*=)"<<endl;
  cout<<(vec_double*=2.0)<<endl;
  cout<<"vector1 div scalar(4): (/)"<<endl;
  cout<<vec_double/4.0<<endl;
  cout<<"vector1 div scalar(4): (/=)"<<endl;
  cout<<(vec_double/=4.0)<<endl;
  cout<<"norm of vector1: "<<endl;
  cout<<vec_double.norm()<<endl;
  cout<<"vector1 after normalization: "<<endl;
  vec_double.normalize();
  cout<<vec_double<<endl;
  return 0;
}
