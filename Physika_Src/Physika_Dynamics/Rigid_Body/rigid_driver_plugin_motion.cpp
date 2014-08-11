/*
 * @file rigid_driver_plugin_motion.h 
 * @Customize the motion of objects
 * @author Tianxiang Zhang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include "Physika_Dynamics/Rigid_Body/rigid_body.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_3d.h"
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin_motion.h"
#include "Physika_Core/Utilities/math_utilities.h"

namespace Physika{


///////////////////////////////////////////////////////////////////////////////////////
//RigidPluginMotionCustomization
///////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
RigidPluginMotionCustomization<Scalar>::RigidPluginMotionCustomization():
    rigid_body_(NULL),
    constant_translation_velocity_(0),
    constant_rotation_velocity_(0),
    period_translation_velocity_(0),
    period_rotation_velocity_(0),
    translation_period_(0),
    rotation_period_(0)
{

}

template <typename Scalar>
RigidPluginMotionCustomization<Scalar>::~RigidPluginMotionCustomization()
{

}

template <typename Scalar>
void RigidPluginMotionCustomization<Scalar>::setRigidBody(RigidBody<Scalar, 3>* rigid_body)
{
    rigid_body_ = rigid_body;
}

template <typename Scalar>
void RigidPluginMotionCustomization<Scalar>::setConstantTranslation(const Vector<Scalar, 3>& velocity)
{
    if(rigid_body_ == NULL)
    {
        std::cerr<<"NULL rigid body in motion customization!"<<std::endl;
        return;
    }
    constant_translation_velocity_ = velocity;
}

template <typename Scalar>
void RigidPluginMotionCustomization<Scalar>::setConstantRotation(const Vector<Scalar, 3>& velocity)
{
    if(rigid_body_ == NULL)
    {
        std::cerr<<"NULL rigid body in motion customization!"<<std::endl;
        return;
    }
    constant_rotation_velocity_ = velocity;
}

template <typename Scalar>
void RigidPluginMotionCustomization<Scalar>::setPeriodTranslation(const Vector<Scalar, 3>& velocity, Scalar period)
{
    if(rigid_body_ == NULL)
    {
        std::cerr<<"NULL rigid body in motion customization!"<<std::endl;
        return;
    }
    if(period <= 0)
    {
        std::cerr<<"Period should be positive in motion customization!"<<std::endl;
        return;
    }
    period_translation_velocity_ = velocity;
    translation_period_ = period;
}

template <typename Scalar>
void RigidPluginMotionCustomization<Scalar>::setPeriodRotation(const Vector<Scalar, 3>& velocity, Scalar period)
{
    if(rigid_body_ == NULL)
    {
        std::cerr<<"NULL rigid body in motion customization!"<<std::endl;
        return;
    }
    if(period <= 0)
    {
        std::cerr<<"Period should be positive in motion customization!"<<std::endl;
        return;
    }
    period_rotation_velocity_ = velocity;
    rotation_period_ = period;
}


template <typename Scalar>
void RigidPluginMotionCustomization<Scalar>::update(Scalar dt, Scalar current_time)
{
    //constant motion
    Vector<Scalar, 3> translation_velocity = constant_translation_velocity_;
    Vector<Scalar, 3> rotation_velocity = constant_rotation_velocity_;

    //period motion
    if(translation_period_ > 0)
    {
        Scalar phase = cos(current_time * 2 * PI / translation_period_);
        translation_velocity += period_translation_velocity_ * phase;
    }
    if(rotation_period_ > 0)
    {
        Scalar phase = cos(current_time * 2 * PI / rotation_period_);
        rotation_velocity += period_rotation_velocity_ * phase;
    }

    //update
    rigid_body_->setGlobalTranslationVelocity(translation_velocity);
    rigid_body_->setGlobalAngularVelocity(rotation_velocity);
    rigid_body_->update(dt, true);
}

///////////////////////////////////////////////////////////////////////////////////////
//RigidDriverPluginMotion
///////////////////////////////////////////////////////////////////////////////////////

template <typename Scalar>
RigidDriverPluginMotion<Scalar>::RigidDriverPluginMotion():
    time_(0)
{

}

template <typename Scalar>
RigidDriverPluginMotion<Scalar>::~RigidDriverPluginMotion()
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onBeginFrame(unsigned int frame)
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onEndFrame(unsigned int frame)
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onBeginTimeStep(Scalar time, Scalar dt)
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onEndTimeStep(Scalar time, Scalar dt)
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onBeginRigidStep(unsigned int step, Scalar dt)
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onEndRigidStep(unsigned int step, Scalar dt)
{
    std::map<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >::iterator map_itr;
    for(map_itr = customized_motions_.begin(); map_itr != customized_motions_.end(); map_itr++)
    {
        (map_itr->second).update(dt, time_);
    }
    time_ += dt;
}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onAddRigidBody(RigidBody<Scalar, 3>* rigid_body)
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onBeginCollisionDetection()
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::onEndCollisionDetection()
{

}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::setConstantTranslation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity)
{
    std::map<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >::iterator map_itr;
    map_itr = customized_motions_.find(rigid_body);
    if(map_itr == customized_motions_.end())
    {
        RigidPluginMotionCustomization<Scalar> customization;
        rigid_body->setFixed(true);
        customization.setRigidBody(rigid_body);
        customization.setConstantTranslation(velocity);
        customized_motions_.insert(std::pair<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >(rigid_body, customization));
    }
    else
    {
        customized_motions_[rigid_body].setConstantTranslation(velocity);
    }
}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::setConstantRotation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity)
{
    std::map<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >::iterator map_itr;
    map_itr = customized_motions_.find(rigid_body);
    if(map_itr == customized_motions_.end())
    {
        RigidPluginMotionCustomization<Scalar> customization;
        rigid_body->setFixed(true);
        customization.setRigidBody(rigid_body);
        customization.setConstantRotation(velocity);
        customized_motions_.insert(std::pair<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >(rigid_body, customization));
    }
    else
    {
        customized_motions_[rigid_body].setConstantRotation(velocity);
    }
}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::setPeriodTranslation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity, Scalar period)
{
    std::map<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >::iterator map_itr;
    map_itr = customized_motions_.find(rigid_body);
    if(map_itr == customized_motions_.end())
    {
        RigidPluginMotionCustomization<Scalar> customization;
        rigid_body->setFixed(true);
        customization.setRigidBody(rigid_body);
        customization.setPeriodTranslation(velocity, period);
        customized_motions_.insert(std::pair<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >(rigid_body, customization));
    }
    else
    {
        customized_motions_[rigid_body].setPeriodTranslation(velocity, period);
    }
}

template <typename Scalar>
void RigidDriverPluginMotion<Scalar>::setPeriodRotation(RigidBody<Scalar, 3>* rigid_body, const Vector<Scalar, 3>& velocity, Scalar period)
{
    std::map<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >::iterator map_itr;
    map_itr = customized_motions_.find(rigid_body);
    if(map_itr == customized_motions_.end())
    {
        RigidPluginMotionCustomization<Scalar> customization;
        rigid_body->setFixed(true);
        customization.setRigidBody(rigid_body);
        customization.setPeriodRotation(velocity, period);
        customized_motions_.insert(std::pair<RigidBody<Scalar, 3>*, RigidPluginMotionCustomization<Scalar> >(rigid_body, customization));
    }
    else
    {
        customized_motions_[rigid_body].setPeriodRotation(velocity, period);
    }
}



//explicit instantiation
template class RigidPluginMotionCustomization<float>;
template class RigidPluginMotionCustomization<double>;
template class RigidDriverPluginMotion<float>;
template class RigidDriverPluginMotion<double>;

}
