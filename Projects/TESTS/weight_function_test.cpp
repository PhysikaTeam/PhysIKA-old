/*
 * @file weight_function_test.cpp 
 * @brief test the weight functions. 
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

#include "Physika_Core/Weight_Functions/weight_function.h"
#include "Physika_Core/Weight_Functions/linear_weight_function.h"
#include "Physika_Core/Weight_Functions/quadratic_weight_function.h"
using namespace std;
using Physika::LinearWeightFunction;
using Physika::QuadraticWeightFunction;
using Physika::WeightFunction;

int main()
{
    WeightFunction<float> *weight_function = new LinearWeightFunction<float>();
    weight_function->printInfo();
    delete weight_function;
    weight_function = new QuadraticWeightFunction<float>();
    weight_function->printInfo();
    delete weight_function;
    return 0;
}
