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

namespace Physika{

/*
 * WeightFunction: Base class of all weight functions.
 * The weight functions are 1D, higer dimension versions
 * could be constructed  with dyadic product of 1D ones.
 * These weight functions are used extensively in numerical
 * methods like SPH.
 */

template <typename Scalar>
class WeightFunction
{
public:
    WeightFunction(){}
    virtual ~WeightFunction(){}
    virtual Scalar weight(Scalar r, Scalar h) const=0; //return the weight at r, given the influence horizon h
    virtual Scalar gradient(Scalar r, Scalar h) const=0;
    virtual void printInfo() const=0;  //print the formula of this weight function
};


} //end of namespace Physika

#endif //PHYSIKA_CORE_WEIGHT_FUNCTIONS_WEIGHT_FUNCTION_H_
