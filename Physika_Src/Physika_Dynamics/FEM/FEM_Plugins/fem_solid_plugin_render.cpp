/*
 * @file fem_solid_plugin_render.cpp 
 * @brief plugin for real-time render of drivers derived from FEMSolid.
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

#include <limits>
#include <iostream>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Geometry/Volumetric_Meshes/volumetric_mesh.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Dynamics/FEM/FEM_Plugins/fem_solid_plugin_render.h"

namespace Physika{

template <typename Scalar, int Dim>
FEMSolidPluginRender<Scalar,Dim>* FEMSolidPluginRender<Scalar,Dim>::active_instance_ = NULL;

template <typename Scalar, int Dim>
FEMSolidPluginRender<Scalar,Dim>::FEMSolidPluginRender()
    :FEMSolidPluginBase<Scalar,Dim>(),
     window_(NULL),pause_simulation_(true),
     simulation_finished_(false),render_velocity_(false),
     velocity_scale_(1.0),auto_capture_frame_(false),
     render_mesh_vertex_(false),render_mesh_wireframe_(true),
     render_mesh_solid_(false),total_time_(0)
{
    activateCurrentInstance();
}

template <typename Scalar, int Dim>
FEMSolidPluginRender<Scalar,Dim>::~FEMSolidPluginRender()
{
    clearVolumetricMeshRender();
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::onBeginFrame(unsigned int frame)
{
//do nothing, because advanceFrame() is never called
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::onEndFrame(unsigned int frame)
{
//do nothing, because advanceFrame() is never called
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::onBeginTimeStep(Scalar time, Scalar dt)
{
    //start timer when the first frame begins
    FEMSolid<Scalar,Dim> *driver = this->driver();
    PHYSIKA_ASSERT(driver);
    Scalar frame_rate = driver->frameRate();
    Scalar cur_frame_scalar = time*frame_rate;
    unsigned int cur_frame = static_cast<unsigned int>(cur_frame_scalar);
    unsigned int start_frame = driver->getStartFrame();
    if(cur_frame_scalar-start_frame<std::numeric_limits<Scalar>::epsilon()) //begins the first frame
    {
        if(driver->isTimerEnabled())
        timer_.startTimer();
        std::cout<<"Begin Frame "<<cur_frame<<"\n";
    }
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::onEndTimeStep(Scalar time, Scalar dt)
{
    //stop timer when a frame ends
    //stop simulation when maximum frame reached
    FEMSolid<Scalar,Dim> *driver = this->driver();
    PHYSIKA_ASSERT(driver);
    unsigned int max_frame = driver->getEndFrame();
    unsigned int start_frame = driver->getStartFrame();
    Scalar frame_rate = driver->frameRate();
    unsigned int cur_frame = static_cast<unsigned int>(time*frame_rate);
    unsigned int frame_last_step = static_cast<unsigned int>((time-dt)*frame_rate);
    if(cur_frame-frame_last_step==1) //ended a frame
    {
        std::cout<<"End Frame "<<cur_frame-1<<" ";
        if(driver->isTimerEnabled())
        {
            timer_.stopTimer();
            Scalar time_cur_frame = timer_.getElapsedTime(); 
            total_time_ += time_cur_frame;
            std::cout<<time_cur_frame<<" s";
        }
        std::cout<<"\n";
        if(this->auto_capture_frame_)  //write screen to file
        {
            std::string file_name("./screen_shots/frame_");
            std::stringstream adaptor;
            adaptor<<cur_frame;
            std::string cur_frame_str;
            adaptor>>cur_frame_str;
            file_name += cur_frame_str+std::string(".png");
            this->window_->saveScreen(file_name);
        }
        if(driver->isTimerEnabled())
        {
            //start timer for next frame
            //not accurate if something happens between time steps
            if(cur_frame < max_frame)
                timer_.startTimer();
        }
    }
    if(cur_frame >= max_frame)
    {
        this->simulation_finished_ = true;
        std::cout<<"Simulation Ended.\n";
        std::cout<<"Total simulation time: "<<total_time_<<" s; Average: "<<total_time_/(max_frame-start_frame+1)<<" s/frame.\n";
    }
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::setDriver(DriverBase<Scalar> *driver)
{
    FEMSolidPluginBase<Scalar,Dim>::setDriver(driver);
    //initialize the volumetric mesh render
    updateVolumetricMeshRender();
}

template <typename Scalar, int Dim>
GlutWindow* FEMSolidPluginRender<Scalar,Dim>::window()
{
    return window_;
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::setWindow(GlutWindow *window)
{
    if(window==NULL)
        throw PhysikaException("NULL window pointer provided to render plugin!");
    window_ = window;
    window_->setIdleFunction(FEMSolidPluginRender<Scalar,Dim>::idleFunction);
    window_->setDisplayFunction(FEMSolidPluginRender<Scalar,Dim>::displayFunction);
    window_->setKeyboardFunction(FEMSolidPluginRender<Scalar,Dim>::keyboardFunction);
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::updateVolumetricMeshRender()
{
    FEMSolid<Scalar,Dim> *fem_driver = this->driver();
    clearVolumetricMeshRender();
    unsigned int obj_num = fem_driver->objectNum();
    volumetric_mesh_render_.resize(obj_num,NULL);
    volumetric_mesh_vert_displacement_.resize(obj_num);
    for(unsigned int obj_idx = 0; obj_idx < obj_num; ++obj_idx)
    {
        volumetric_mesh_vert_displacement_[obj_idx].resize(fem_driver->numSimVertices(obj_idx));
        const VolumetricMesh<Scalar,Dim> &sim_mesh = fem_driver->simulationMesh(obj_idx);
        volumetric_mesh_render_[obj_idx] = new VolumetricMeshRender<Scalar,Dim>(&sim_mesh,&volumetric_mesh_vert_displacement_[obj_idx]);
        volumetric_mesh_render_[obj_idx]->disableDisplayList();
    }
}


template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::idleFunction(void)
{
    PHYSIKA_ASSERT(active_instance_);
    FEMSolid<Scalar,Dim> *driver = active_instance_->driver();
    PHYSIKA_ASSERT(driver);
    Scalar dt = driver->computeTimeStep();
    if(active_instance_->pause_simulation_ == false &&
       active_instance_->simulation_finished_ == false)
    {
        driver->advanceStep(dt);
    }
    glutPostRedisplay();
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::displayFunction(void)
{
    PHYSIKA_ASSERT(active_instance_);
    GlutWindow *window = active_instance_->window_;
    PHYSIKA_ASSERT(window);
    Color<double> background_color = window->backgroundColor<double>();
    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    window->applyCameraAndLights();

    //render vertex velocity
    if(active_instance_->render_velocity_)
        active_instance_->renderVertexVelocity();
    //render deformed volumetric mesh
    active_instance_->updateVolumetricMeshDisplacement();
    for(unsigned int i = 0; i < active_instance_->volumetric_mesh_render_.size(); ++i)
    {
        if(active_instance_->render_mesh_solid_)
            active_instance_->volumetric_mesh_render_[i]->enableRenderSolid();
        else
            active_instance_->volumetric_mesh_render_[i]->disableRenderSolid();
        if(active_instance_->render_mesh_vertex_)
            active_instance_->volumetric_mesh_render_[i]->enableRenderVertices();
        else
            active_instance_->volumetric_mesh_render_[i]->disableRenderVertices();
        if(active_instance_->render_mesh_wireframe_)
            active_instance_->volumetric_mesh_render_[i]->enableRenderWireframe();
        else
            active_instance_->volumetric_mesh_render_[i]->disableRenderWireframe();
        active_instance_->volumetric_mesh_render_[i]->render();
    }
    
    (window->renderManager()).renderAll(); //render all other tasks of render manager
    window->displayFrameRate();
    glutSwapBuffers();
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::keyboardFunction(unsigned char key, int x, int y)
{
    PHYSIKA_ASSERT(active_instance_);
    GlutWindow *window = active_instance_->window_;
    PHYSIKA_ASSERT(window);
    window->bindDefaultKeys(key,x,y);  //default key is preserved
    switch(key)
    {
    case 32: //space
        active_instance_->pause_simulation_ = !(active_instance_->pause_simulation_);
    case 'i':
        active_instance_->velocity_scale_ *= 2.0;
        break;
    case 'd':
        active_instance_->velocity_scale_ /= 2.0;
        break;
    case 'v':
        active_instance_->render_velocity_ = !(active_instance_->render_velocity_);
        break;
    case 'S':
        active_instance_->auto_capture_frame_ = !(active_instance_->auto_capture_frame_);
        break;
    case 'p':
        active_instance_->render_mesh_vertex_ = !(active_instance_->render_mesh_vertex_);
        break;
    case 'w':
        active_instance_->render_mesh_wireframe_ = !(active_instance_->render_mesh_wireframe_);
        break;
    case 'm':
        active_instance_->render_mesh_solid_ = !(active_instance_->render_mesh_solid_);
        break;
    default:
        break;
    }
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::activateCurrentInstance()
{
    active_instance_ = this;
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::renderVertexVelocity()
{
    FEMSolid<Scalar,Dim> *fem_driver = this->driver();
    PHYSIKA_ASSERT(fem_driver);
    openGLColor3(Color<Scalar>::Red());
    glDisable(GL_LIGHTING);
    for(unsigned int obj_idx = 0; obj_idx < fem_driver->objectNum(); ++obj_idx)
    {
        for(unsigned int vert_idx = 0; vert_idx < fem_driver->numSimVertices(obj_idx); ++vert_idx)
        {
            Vector<Scalar,Dim> start = fem_driver->vertexCurrentPosition(obj_idx,vert_idx);
            Vector<Scalar,Dim> end = start + (velocity_scale_)*(fem_driver->vertexVelocity(obj_idx,vert_idx));
            glBegin(GL_LINES);
            openGLVertex(start);
            openGLVertex(end);
            glEnd();
        }
    }
}

template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::clearVolumetricMeshRender()
{
    for(unsigned int i = 0; i < volumetric_mesh_render_.size(); ++i)
        if(volumetric_mesh_render_[i])
            delete volumetric_mesh_render_[i];
    volumetric_mesh_render_.clear();
}
    
template <typename Scalar, int Dim>
void FEMSolidPluginRender<Scalar,Dim>::updateVolumetricMeshDisplacement()
{
    FEMSolid<Scalar,Dim> *fem_driver = this->driver();
    PHYSIKA_ASSERT(fem_driver);
    PHYSIKA_ASSERT(volumetric_mesh_vert_displacement_.size() == fem_driver->objectNum());
    PHYSIKA_ASSERT(volumetric_mesh_render_.size() == fem_driver->objectNum());
    for(unsigned int obj_idx = 0; obj_idx < fem_driver->objectNum(); ++obj_idx)
    {
        PHYSIKA_ASSERT(volumetric_mesh_vert_displacement_[obj_idx].size() == fem_driver->numSimVertices(obj_idx));
        for(unsigned int vert_idx = 0; vert_idx < fem_driver->numSimVertices(obj_idx); ++vert_idx)
            volumetric_mesh_vert_displacement_[obj_idx][vert_idx] = fem_driver->vertexDisplacement(obj_idx,vert_idx);
    }
}

//explicit instantiations
template class FEMSolidPluginRender<float,2>;
template class FEMSolidPluginRender<float,3>;
template class FEMSolidPluginRender<double,2>;
template class FEMSolidPluginRender<double,3>;

}  //end of namespace Physika
