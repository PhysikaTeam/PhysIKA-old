/*
 * @file weight_function_creator.h 
 * @brief weight function creator. 
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_CORE_WEIGHT_FUNCTIONS_WEIGHT_FUNCTION_CREATOR_H_
#define PHYSIKA_CORE_WEIGHT_FUNCTIONS_WEIGHT_FUNCTION_CREATOR_H_

namespace Physika{

template <typename Scalar, int Dim> class WeightFunction;

template <typename WeightFunctionTypeName>
class WeightFunctionCreator
{
public:
    static WeightFunction<typename WeightFunctionTypeName::ScalarType,WeightFunctionTypeName::DimSize>* createWeightFunction()
    {
        return new WeightFunctionTypeName();
    }
};

}  //end of namespace Physika

#endif
