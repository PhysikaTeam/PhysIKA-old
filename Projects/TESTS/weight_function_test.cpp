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

#include "Physika_Core/Weight_Functions/weight_function_creator.h"
#include "Physika_Core/Weight_Functions/weight_function.h"
#include "Physika_Core/Weight_Functions/linear_weight_functions.h"
#include "Physika_Core/Weight_Functions/quadratic_weight_functions.h"
#include "Physika_Core/Weight_Functions/cubic_weight_functions.h"
#include "Physika_Core/Weight_Functions/quartic_weight_functions.h"
using namespace std;
using Physika::LinearWeightFunction;
using Physika::JohnsonQuadraticWeightFunction;
using Physika::DomeShapedQuadraticWeightFunction;
using Physika::PiecewiseCubicSpline;
using Physika::LucyQuarticWeightFunction;
using Physika::NewQuarticWeightFunction;
using Physika::WeightFunction;
using Physika::WeightFunctionCreator;
using Physika::DesbrunSpikyWeightFunction;

int main()
{
    //linear weight function
    WeightFunction<float,1> *weight_function_1d = new LinearWeightFunction<float,1>();
    WeightFunction<float,2> *weight_function_2d = new LinearWeightFunction<float,2>();
    WeightFunction<float,3> *weight_function_3d = new LinearWeightFunction<float,3>();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    //johnson quadratic weight function
    weight_function_1d = new JohnsonQuadraticWeightFunction<float,1>();
    weight_function_2d = new JohnsonQuadraticWeightFunction<float,2>();
    weight_function_3d = new JohnsonQuadraticWeightFunction<float,3>();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    //Dome-shaped quadratic weight function
    weight_function_1d = new DomeShapedQuadraticWeightFunction<float,1>();
    weight_function_2d = new DomeShapedQuadraticWeightFunction<float,2>();
    weight_function_3d = new DomeShapedQuadraticWeightFunction<float,3>();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    //piece-wise cubic spline
    weight_function_1d = new PiecewiseCubicSpline<float,1>();
    weight_function_2d = new PiecewiseCubicSpline<float,2>();
    weight_function_3d = new PiecewiseCubicSpline<float,3>();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    //piece-wise cubic spline
    weight_function_1d = new LucyQuarticWeightFunction<float,1>();
    weight_function_2d = new LucyQuarticWeightFunction<float,2>();
    weight_function_3d = new LucyQuarticWeightFunction<float,3>();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    //piece-wise cubic spline
    weight_function_1d = new NewQuarticWeightFunction<float,1>();
    weight_function_2d = new NewQuarticWeightFunction<float,2>();
    weight_function_3d = new NewQuarticWeightFunction<float,3>();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    //weight function creator
    weight_function_1d = WeightFunctionCreator<DesbrunSpikyWeightFunction<float,1>>::createWeightFunction();
    weight_function_2d = WeightFunctionCreator<DesbrunSpikyWeightFunction<float,2>>::createWeightFunction();
    weight_function_3d = WeightFunctionCreator<DesbrunSpikyWeightFunction<float,3>>::createWeightFunction();
    weight_function_1d->printInfo();
    weight_function_2d->printInfo();
    weight_function_3d->printInfo();
    delete weight_function_1d;
    delete weight_function_2d;
    delete weight_function_3d;
    return 0;
}
