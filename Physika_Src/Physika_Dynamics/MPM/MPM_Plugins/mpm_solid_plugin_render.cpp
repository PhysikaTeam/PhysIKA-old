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
#include <vector>
#include <GL/freeglut.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Dynamics/Particles/solid_particle.h"
#include "Physika_Dynamics/MPM/MPM_Plugins/mpm_solid_plugin_render.h"

namespace Physika{

template <typename Scalar, int Dim>
MPMSolidPluginRender<Scalar,Dim>* MPMSolidPluginRender<Scalar,Dim>::active_instance_ = NULL;

template <typename Scalar, int Dim>
MPMSolidPluginRender<Scalar,Dim>::MPMSolidPluginRender()
    :MPMSolidPluginBase<Scalar,Dim>(),window_(NULL),
     render_particle_(true),render_grid_(true),
     render_particle_velocity_(false),render_grid_velocity_(false)
{
    activateCurrentInstance();
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
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onSolveOnGrid(Scalar dt)
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onPerformGridCollision(Scalar dt)
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onPerformParticleCollision(Scalar dt)
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticleInterpolationWeight()
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticleConstitutiveModelState(Scalar dt)
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticleVelocity()
{
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::onUpdateParticlePosition(Scalar dt)
{
}

template <typename Scalar, int Dim>
GlutWindow* MPMSolidPluginRender<Scalar,Dim>::window()
{
    return window_;
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::setWindow(GlutWindow *window)
{
    if(window==NULL)
    {
        std::cerr<<"Error: NULL window pointer provided to render plugin, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    window_ = window;
    window_->setIdleFunction(MPMSolidPluginRender<Scalar,Dim>::idleFunction);
    window_->setDisplayFunction(MPMSolidPluginRender<Scalar,Dim>::displayFunction);
    window_->setKeyboardFunction(MPMSolidPluginRender<Scalar,Dim>::keyboardFunction);
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::idleFunction(void)
{
//TO DO
    glutPostRedisplay();
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::displayFunction(void)
{
    PHYSIKA_ASSERT(active_instance_);
    GlutWindow *window = active_instance_->window_;
    PHYSIKA_ASSERT(window);
    Color<double> background_color = window->backgroundColor<double>();
    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    window->applyCameraAndLights();

    if(active_instance_->render_particle_)
        active_instance_->renderParticles();
    if(active_instance_->render_grid_)
        active_instance_->renderGrid();
    if(active_instance_->render_particle_velocity_)
        active_instance_->renderParticleVelocity();
    if(active_instance_->render_grid_velocity_)
        active_instance_->renderGridVelocity();
    window->disableDisplayFrameRate();
    glutSwapBuffers();
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::keyboardFunction(unsigned char key, int x, int y)
{
    PHYSIKA_ASSERT(active_instance_);
    GlutWindow *window = active_instance_->window_;
    PHYSIKA_ASSERT(window);
    window->bindDefaultKeys(key,x,y);  //default key is preserved
    switch(key)
    {
    case 'p':
        active_instance_->render_particle_ = !(active_instance_->render_particle_);
        break;
    case 'g':
        active_instance_->render_grid_ = !(active_instance_->render_grid_);
        break;
    case 'P':
        active_instance_->render_particle_velocity_ = !(active_instance_->render_particle_velocity_);
        break;
    case 'G':
        active_instance_->render_grid_velocity_ = !(active_instance_->render_grid_velocity_);
        break;
    default:
        break;
    }
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::activateCurrentInstance()
{
    active_instance_ = this;
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::renderParticles()
{
    PHYSIKA_ASSERT(active_instance_);
    MPMSolid<Scalar,Dim> *driver = active_instance_->driver();
    PHYSIKA_ASSERT(driver);
    std::vector<SolidParticle<Scalar,Dim>*> particles;
    driver->allParticles(particles);
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::renderGrid()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::renderParticleVelocity()
{
//TO DO
}

template <typename Scalar, int Dim>
void MPMSolidPluginRender<Scalar,Dim>::renderGridVelocity()
{
//TO DO
}

//explicit instantiations
template class MPMSolidPluginRender<float,2>;
template class MPMSolidPluginRender<float,3>;
template class MPMSolidPluginRender<double,2>;
template class MPMSolidPluginRender<double,3>;

}  //end of namespace Physika
