/*
 * @file  constitutive_model.h
 * @brief Constitutive model of deformable solids, abstract class
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

#ifndef PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_H_
#define PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_H_

namespace Physika{

class ConstitutiveModel
{
public:
    ConstitutiveModel(){}
    virtual ~ConstitutiveModel(){}
    virtual void info() const=0;
protected:
};

}  //end of namespace Physika

#endif //PHYSIKA_DYNAMICS_CONSTITUTIVE_MODELS_CONSTITUTIVE_MODEL_H_
