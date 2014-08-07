/*
 * @file weight_function.h 
 * @brief base class of all weight functions, abstract class. 
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

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_WEIGHT_FUNCTION_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_WEIGHT_FUNCTION_H_

#include "Physika_Core/Vectors/vector_2d.h"
#include "Physika_Core/Vectors/vector_3d.h"

namespace Physika{

/*
 * WeightFunction: Base class of all weight functions.
 * The weight functions are functions of the distance
 * to the center of the functions' support domain.
 * The support domain of a weight function is:
 * 1. an interval in 1D with length 2*R
 * 2. a circle in 2D with radius R
 * 3. a sphere in 3D with radius R
 * R is the radius of the support domain.
 *
 * Generally a weight function satisfies following properties:
 * 1. normalized over support domain
 * 2. compactly supported: f(x,R)=0, for |x|>R
 * 3. Postive in support domain
 * 4. sufficiently smooth
 * 5. monotonically decreasing with larger distance
 * 6. etc
 *
 */


template <typename Scalar, int Dim>
class WeightFunction
{
public:
    WeightFunction(){}
    virtual ~WeightFunction(){}
    virtual Scalar weight(const Vector<Scalar,Dim> &center_to_x, Scalar R) const=0; 
    virtual Vector<Scalar,Dim> gradient(const Vector<Scalar,Dim> &center_to_x, Scalar R) const=0;
    virtual Scalar laplacian(const Vector<Scalar,Dim> &center_to_x, Scalar R) const=0;
    virtual void printInfo() const=0;  //print the formula of this weight function

    typedef Scalar ScalarType;
    static const int DimSize = Dim;
};

/*
 * partial sepcialization for 1D as the arguments ar Scalar
 */

template <typename Scalar>
class WeightFunction<Scalar,1>
{
public:
    WeightFunction(){}
    virtual ~WeightFunction(){}
    virtual Scalar weight(Scalar center_to_x, Scalar R) const=0; 
    virtual Scalar gradient(Scalar center_to_x, Scalar R) const=0;
    virtual Scalar laplacian(Scalar center_to_x, Scalar R) const=0;
    virtual void printInfo() const=0;  //print the formula of this weight function

    typedef Scalar ScalarType;
    static const int DimSize = 1;
};

} //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_WEIGHT_FUNCTION_H_
