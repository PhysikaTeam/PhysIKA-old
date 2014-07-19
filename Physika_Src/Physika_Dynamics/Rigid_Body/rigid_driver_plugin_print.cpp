/*
 * @file rigid_driver_plugin_print.cpp 
 * @Print plugin of rigid body driver. Print information during simulation
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
#include "Physika_Dynamics/Rigid_Body/rigid_body_driver.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin.h"
#include "Physika_Dynamics/Rigid_Body/rigid_driver_plugin_print.h"

namespace Physika{

template <typename Scalar,int Dim>
RigidDriverPluginPrint<Scalar, Dim>::RigidDriverPluginPrint()
{

}

template <typename Scalar,int Dim>
RigidDriverPluginPrint<Scalar, Dim>::~RigidDriverPluginPrint()
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onBeginFrame(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onEndFrame(unsigned int frame)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onBeginTimeStep(Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onEndTimeStep(Scalar time, Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onBeginRigidStep(unsigned int step, Scalar dt)
{
    std::cout<<"Frame begin: "<<step<<std::endl;
    //std::cout<<this->rigid_driver_->rigidBody(0)->globalMassCenter()<<std::endl;
    //std::cout<<this->rigid_driver_->rigidBody(0)->globalTranslation()<<std::endl;
    //std::cout<<this->rigid_driver_->rigidBody(0)->transform().scale()<<std::endl;
    //std::cout<<this->rigid_driver_->rigidBody(0)->transform().translation()<<std::endl;
    //std::cout<<this->rigid_driver_->rigidBody(0)->transform().rotation().getEulerAngle()<<std::endl;


}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onEndRigidStep(unsigned int step, Scalar dt)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onAddRigidBody(RigidBody<Scalar, Dim>* rigid_body)
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onBeginCollisionDetection()
{

}

template <typename Scalar,int Dim>
void RigidDriverPluginPrint<Scalar, Dim>::onEndCollisionDetection()
{
    std::cout<<"PCS: "<<this->rigid_driver_->collisionResult().numberPCS()<<std::endl;
    std::cout<<"Collision: "<<this->rigid_driver_->collisionResult().numberCollision()<<std::endl;
    std::cout<<"Contact: "<<this->rigid_driver_->contactPoints().numContactPoint()<<std::endl;
}


//explicit instantiation
template class RigidDriverPluginPrint<float, 3>;
template class RigidDriverPluginPrint<double, 3>;

}
