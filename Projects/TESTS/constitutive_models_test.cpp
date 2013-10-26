/*
 * @file constitutive_models_test.cpp 
 * @brief Test the constitutive models of solids.
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
#include "Physika_Core/Matrices/matrix_3x3.h"
#include "Physika_Dynamics/Constitutive_Models/neo_hookean.h"
using namespace std;
using Physika::NeoHookean;
using Physika::Matrix2x2;
using Physika::Matrix3x3;

int main()
{
    NeoHookean<float,3> neo_hookean_material;
    Matrix3x3<float> F;
    F.derived();
    neo_hookean_material.info();
    cout<<neo_hookean_material.energy(F)<<endl;
    return 0;
}
