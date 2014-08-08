/*
 * @file mpm_solid_plugin_render.cpp 
 * @brief plugin for real-time render of MPMSolid driver.
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

#include <cstdlib>
#include <iostream>
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_render.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidPluginRender<Scalar,Dim>::MPMSolidPluginRender()
    :MPMSolidPluginBase<Scalar,Dim>(),window_(NULL)
{
}

template <typename Scalar, int Dim>
MPMSolidPluginRender<Scalar,Dim>::~MPMSolidPluginRender()
{
}


template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onBeginFrame(unsigned int frame)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onEndFrame(unsigned int frame)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onBeginTimeStep(Scalar time, Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onEndTimeStep(Scalar time, Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
MPMSolid<Scalar,Dim>* MPMSolidPluginRender<Scalar,Dim>::driver()
{
    MPMSolid<Scalar,Dim> *driver = dynamic_cast<MPMSolid<Scalar,Dim>*>(this->driver_);
    return driver;
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::setDriver(DriverBase<Scalar> *driver)
{
    if(driver==NULL)
    {
        std::cerr<<"Error: NULL driver pointer provided to driver plugin, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    if(dynamic_cast<MPMSolid<Scalar,Dim>*>(driver)==NULL)
    {
        std::cerr<<"Error: Wrong type of driver specified, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->driver_ = driver;
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onRasterize()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onSolveOnGrid(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onPerformGridCollision(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onPerformParticleCollision(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticleInterpolationWeight()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticleConstitutiveModelState(Scalar dt)
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticleVelocity()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticlePosition(Scalar dt)
{
//TO DO
}

//explicit instantiations
template class MPMSolidPluginRender<float,2>;
template class MPMSolidPluginRender<float,3>;
template class MPMSolidPluginRender<double,2>;
template class MPMSolidPluginRender<double,3>;

}  //end of namespace Physika
