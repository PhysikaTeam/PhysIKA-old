/*
 * @file quartic_weight_functions.h 
 * @brief collection of quartic weight functions. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUARTIC_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUARTIC_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * LucyQuarticWeightFunction:
 * reference: "Numerical approach to testing the fission hypothesis"
 * f(x,R) = a*(1+3*(|x|/R))*(1-|x|/R)^3,  0 <= |x| <= R
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class LucyQuarticWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    LucyQuarticWeightFunction(){}
    ~LucyQuarticWeightFunction(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    void printInfo() const;
};

template <typename Scalar>
class LucyQuarticWeightFunction<Scalar,1>: public WeightFunction<Scalar,1>
{
public:
    LucyQuarticWeightFunction(){}
    ~LucyQuarticWeightFunction(){}
    Scalar weight(Scalar center_to_x, Scalar R) const;
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;
};

/*
 * NewQuarticWeightFunction:
 * reference: "A general approach for constructing smoothing function for meshfree methods"
 * let R = 2*h
 * f(x,h) = a*(2/3-9/8*(|x|/h)^2+19/24*(|x|/h)^3-5/32*(|x|/h)^4) (0 <= |x| <= 2h)
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class NewQuarticWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    NewQuarticWeightFunction(){}
    ~NewQuarticWeightFunction(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    void printInfo() const;
};

template <typename Scalar>
class NewQuarticWeightFunction<Scalar,1>: public WeightFunction<Scalar,1>
{
public:
    NewQuarticWeightFunction(){}
    ~NewQuarticWeightFunction(){}
    Scalar weight(Scalar center_to_x, Scalar R) const;
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUARTIC_WEIGHT_FUNCTIONS_H_
