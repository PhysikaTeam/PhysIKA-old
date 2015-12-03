/*
 * @file mpm_solid_plugin_print.cpp 
 * @brief plugin to print information on screen for drivers derived from MPMSolidBase.
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

#include <iostream>
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_print.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidPluginPrint<Scalar,Dim>::MPMSolidPluginPrint()
    :MPMSolidPluginBase<Scalar,Dim>()
{
}

template <typename Scalar, int Dim>
MPMSolidPluginPrint<Scalar,Dim>::~MPMSolidPluginPrint()
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onBeginFrame(unsigned int frame)
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onEndFrame(unsigned int frame)
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onBeginTimeStep(Scalar time, Scalar dt)
{
    std::cout<<"Begin time step, time: "<<time<<", time step: "<<dt<<"\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onEndTimeStep(Scalar time, Scalar dt)
{
    std::cout<<"End time step, time: "<<time<<", time step: "<<dt<<"\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar, Dim>::setDriver(DriverBase<Scalar>* driver)
{
    if (driver == NULL)
        throw PhysikaException("Error: NULL driver pointer provided to driver plugin!");
    MPMSolidBase<Scalar, Dim> *mpm_driver = dynamic_cast<MPMSolidBase<Scalar, Dim>*>(driver);
    if (mpm_driver == NULL)
        throw PhysikaException("Wrong type of driver specified!");
    else
        mpm_driver->enableSolverStatusLog();
    this->driver_ = mpm_driver;
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onRasterize()
{
    std::cout<<"Rasterize particle data to grid.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onSolveOnGrid(Scalar dt)
{
    std::cout<<"Solve momentum equation on grid.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onResolveContactOnGrid(Scalar dt)
{
    std::cout<<"Resolve contact on grid.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onResolveContactOnParticles(Scalar dt)
{
    std::cout<<"Resolve contact on particles.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onUpdateParticleInterpolationWeight()
{
    std::cout<<"Update particle interpolation weight.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onUpdateParticleConstitutiveModelState(Scalar dt)
{
    std::cout<<"Update particle constitutive model.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onUpdateParticleVelocity()
{
    std::cout<<"Update particle velocity with grid velocity.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onApplyExternalForceOnParticles(Scalar dt)
{
    std::cout<<"Apply external force on particles.\n";
}

template <typename Scalar, int Dim>
void MPMSolidPluginPrint<Scalar,Dim>::onUpdateParticlePosition(Scalar dt)
{
    std::cout<<"Update particle position.\n";
}

//explicit instantiations
template class MPMSolidPluginPrint<float,2>;
template class MPMSolidPluginPrint<float,3>;
template class MPMSolidPluginPrint<double,2>;
template class MPMSolidPluginPrint<double,3>;

}  //end of namespace Physika
