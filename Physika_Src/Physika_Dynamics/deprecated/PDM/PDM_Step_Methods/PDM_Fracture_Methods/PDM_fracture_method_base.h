/*
 * @file PDM_fracture_method_base.h 
 * @brief base class of fracture method for PDM drivers.
 * @author Wei Chen
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_FRACTURE_METHODS_PDM_FRACTURE_METHOD_BASE_H
#define PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_FRACTURE_METHODS_PDM_FRACTURE_METHOD_BASE_H

#include <vector>
#include <list>

namespace Physika{

template <typename Scalar, int Dim> class PDMBase;
template <typename Scalar, int Dim> class PDMFamily;

template <typename Scalar, int Dim>
class PDMFractureMethodBase
{
public:
    PDMFractureMethodBase();
    PDMFractureMethodBase(Scalar critical_s);
    virtual ~PDMFractureMethodBase();

    void setHomogeneousCriticalStretch(Scalar critical_s);
    void setCriticalStretch(unsigned int par_idx, Scalar critical_s);
    void setCriticalStretchVec(const std::vector<Scalar> & critical_s_vec);
    void setDriver(PDMBase<Scalar,Dim> * driver);
    void setAlpha(Scalar alpha);
    void setEnhancedStretchTimes(Scalar enhanced_s_times);

    Scalar criticalStretch(unsigned int par_idx) const;
    Scalar alpha() const;

    // core virtual function
    virtual bool applyFracture(Scalar s, unsigned int par_idx, std::list<PDMFamily<Scalar, Dim> > & family, typename std::list<PDMFamily<Scalar,Dim> >::iterator test_par_iter);

protected:
    std::vector<Scalar> critical_s_vec_; //default: 0.2
    PDMBase<Scalar,Dim> * driver_;
    Scalar alpha_;             // default:0.0, user-specified constant to model the fact that material become stronger under compression
    Scalar enhanced_s_times_;  // default:0.0, paramters used to increase the diffcultly of fracture when some fractures already happend
};

}// end of namespace Physika

#endif //PHYSIKA_DYNAMICS_PDM_PDM_STEP_METHODS_PDM_FRACTURE_METHOD_PDM_FRACTURE_METHOD_BASE_H