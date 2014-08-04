/*
 * @file quadratic_weight_functions.h 
 * @brief collection of quadratic weight functions. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUADRATIC_WEIGHT_FUNCTIONS_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUADRATIC_WEIGHT_FUNCTIONS_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"
#include "Physika_Core/Weight_Functions/weight_function.h"

namespace Physika{

/*
 * JohnsonQuadraticWeightFunction: 
 * reference: "SPH for high velocity impact computations"
 * let h = 0.5*R,
 * f(x,h) = a*(3/16*(|x|/h)^2-3/4*(|x|/h)+3/4) (0 <= |x| <= 2*h)
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class JohnsonQuadraticWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    JohnsonQuadraticWeightFunction(){}
    ~JohnsonQuadraticWeightFunction(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    void printInfo() const;
};

template <typename Scalar>
class JohnsonQuadraticWeightFunction<Scalar,1>: public WeightFunction<Scalar,1>
{
public:
    JohnsonQuadraticWeightFunction(){}
    ~JohnsonQuadraticWeightFunction(){}
    Scalar weight(Scalar center_to_x, Scalar R) const;
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;
};

/*
 * DomeShapedQuadraticWeightFunction:
 * reference: "Lanczo's generalized derivative: insights and applications"
 * f(x,R) = a*(1-(|x|/R)^2) (0 <= |x| <= R)
 * where 'a' depends on the dimension and radius of support domain
 */

template <typename Scalar, int Dim>
class DomeShapedQuadraticWeightFunction: public WeightFunction<Scalar,Dim>
{
public:
    DomeShapedQuadraticWeightFunction(){}
    ~DomeShapedQuadraticWeightFunction(){}
    Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const;
    void printInfo() const;
};

template <typename Scalar>
class DomeShapedQuadraticWeightFunction<Scalar,1>: public WeightFunction<Scalar,1>
{
public:
    DomeShapedQuadraticWeightFunction(){}
    ~DomeShapedQuadraticWeightFunction(){}
    Scalar weight(Scalar center_to_x, Scalar R) const;
    Scalar gradient(Scalar center_to_x, Scalar R) const;
    Scalar laplacian(Scalar center_to_x, Scalar R) const;
    void printInfo() const;
};

}  //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_QUADRATIC_WEIGHT_FUNCTIONS_H_
