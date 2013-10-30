/*
 * @file quaternion_test.cpp 
 * @brief Test quaternion_test class.
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
#include "Physika_Core/Quaternion/quaternion.h"
#include "Physika_Core/Utilities/global_config.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Array/array.h"
using namespace std;
using Physika::Quaternion;
using Physika::Quaternionf;
using Physika::Quaterniond;
using Physika::Array;

//template <typename Scalar>
//std::ostream& operator<< (std::ostream &s, const Quaternion<Scalar> &quat)
//{
//    return s; 
//}

int main()
{
    cout<<"Quaternion Test:"<<endl;

    Quaterniond quata;
    cout<<"Quata: \n"<<quata;

    Quaterniond quatb(2,2,2,1);
    cout<<"Quatb: \n"<<quatb;

    quata += quatb;
    cout<<"Quata += Quatb :\n"<<quata;

    quata -= quatb;
    cout<<"Quata -= Quatb :\n"<<quatb;

    cout<<"Quatb's norm: \n"<<quatb.norm()<<endl;

    quatb.normalize();
    cout<<"Quatb's normalize: \n"<<quatb;
    
	int a;
	cin>>a;
}
