/*
 * @file PDM_plugin_render.cpp 
 * @brief render plugins class for PDM drivers.
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

#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <GL/gl.h>
#include <GL/freeglut.h>
#include "Physika_Dynamics/PDM/PDM_Plugins/PDM_plugin_render.h"
#include "Physika_GUI/Glut_Window/glut_window.h"
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Point_Render/point_render.h"
#include "Physika_Render/Volumetric_Mesh_Render/volumetric_mesh_render.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Render/Color/color.h"
#include "Physika_Dynamics/PDM/PDM_base.h"
#include "Physika_Dynamics/PDM/PDM_particle.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_base.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_step_method_state_viscoplasticity.h"
#include "Physika_Dynamics/PDM/PDM_Step_Methods/PDM_Impact_Methods/PDM_impact_method_base.h"

namespace Physika{

template <typename Scalar, int Dim>
PDMPluginRender<Scalar, Dim> * PDMPluginRender<Scalar, Dim>::active_instance_ = NULL;

template <typename Scalar, int Dim>
PDMPluginRender<Scalar, Dim>::PDMPluginRender()
    :PDMPluginBase(),window_(NULL),point_size_(1.0),
    render_particle_(true),render_specified_family_(false),
    render_velocity_(false),auto_capture_frame_(false),floor_pos_(0.0),render_floor_(false),
    render_bullet_(false),impact_method_(NULL),bullet_radius_(4.0),bullet_pos_(0),bullet_velocity_(0),render_s_(false),
    render_cur_size_(false),render_mesh_(false),
    render_compression_plane_(false), up_plane_pos_(0.0), down_plane_pos_(0.0), up_plane_vel_(0.0), down_plane_vel_(0.0)
{
    this->activateCurrentInstance();
}

template <typename Scalar, int Dim>
PDMPluginRender<Scalar, Dim>::~PDMPluginRender()
{

}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::onBeginFrame(unsigned int frame)
{
    // to do
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::onEndFrame(unsigned int frame)
{
    if(this->auto_capture_frame_)
    {
        std::string file_name("./screen_shots/frame_");
        std::stringstream adaptor;
        adaptor<<frame;
        std::string cur_frame_str;
        adaptor>>cur_frame_str;
        file_name += cur_frame_str + std::string(".png");
        this->window_->saveScreen(file_name);
    }
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::onBeginTimeStep(Scalar time, Scalar dt)
{
    PDMBase<Scalar, Dim> * driver = this->driver();

    if (driver->isSimulationPause() == false)
    {
        if (this->render_bullet_ && this->impact_method_ == NULL)
            this->bullet_pos_ += this->bullet_velocity_*dt;

        //////////////////////////////////////////////////////////////////////////////////////////
        std::fstream file("compression_plane.txt", std::ios::in);
        if (file.fail() == false)
        {
            std::string parameter_name;

            Scalar up_plane_pos;
            Scalar down_plane_pos;
            unsigned int convert_step;
            Scalar first_up_plane_vel;
            Scalar second_up_plane_vel;
            Scalar first_down_plane_vel;
            Scalar second_down_plane_vel;

            file>>parameter_name>>this->render_compression_plane_;    
            file>>parameter_name>>up_plane_pos;
            file>>parameter_name>>down_plane_pos;
            file>>parameter_name>>convert_step;
            file>>parameter_name>>first_up_plane_vel;
            file>>parameter_name>>second_up_plane_vel;
            file>>parameter_name>>first_down_plane_vel;
            file>>parameter_name>>second_down_plane_vel;

            PDMBase<Scalar, Dim> * pdm_base = this->driver();

            if (pdm_base->timeStepId() == 0)
            {
                this->up_plane_pos_ = up_plane_pos;
                this->down_plane_pos_ = down_plane_pos;
            }

            if (pdm_base->timeStepId() <= convert_step)
            {
                this->up_plane_vel_ = first_up_plane_vel;
                this->down_plane_vel_ = first_down_plane_vel;
            }
            else
            {
                this->up_plane_vel_ = second_up_plane_vel;
                this->down_plane_vel_ = second_down_plane_vel;
            }
        }
        file.close();
        /////////////////////////////////////////////////////////////////////////////////////////

        if (this->render_compression_plane_)
        {
            //update up and down plane pos
            this->up_plane_pos_ += this->up_plane_vel_*dt;
            this->down_plane_pos_ += this->down_plane_vel_*dt;
        }
        
    }
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::onEndTimeStep(Scalar time, Scalar dt)
{
    glutMainLoopEvent();
    glutPostRedisplay();
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableCaptureScreen()
{
    this->auto_capture_frame_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableCaptureScreen()
{
    this->auto_capture_frame_ = false;
}

template <typename Scalar, int Dim>
GlutWindow * PDMPluginRender<Scalar, Dim>::window()
{
    return this->window_;
}

template <typename Scalar, int Dim>
Scalar PDMPluginRender<Scalar, Dim>::pointSize() const
{
    return  this->point_size_;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setPointSize(Scalar point_size)
{
    if(point_size<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"point size too small, default size 1 is used\n";
        point_size = 1.0;
    }
    
    this->point_size_ = point_size;
}

template <typename Scalar, int Dim>
Scalar PDMPluginRender<Scalar, Dim>::velocityScale() const
{
    return this->velocity_scale_;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setVelocityScale(Scalar velocity_scale)
{
    if(velocity_scale<std::numeric_limits<Scalar>::epsilon())
    {
        std::cerr<<"point size too small, default size 1 is used\n";
        velocity_scale = 1.0;
    }

    this->velocity_scale_ = velocity_scale;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setFloorPos(Scalar pos)
{
    this->floor_pos_ = pos;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setImpactMethod(PDMImpactMethodBase<Scalar, Dim> * impact_method)
{
    this->impact_method_ = impact_method;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setBulletPos(const Vector<Scalar,Dim> & bullet_pos)
{
    this->bullet_pos_ = bullet_pos;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setBulletRadius(Scalar bullet_radius)
{
    this->bullet_radius_ = bullet_radius;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setBulletVelocity(const Vector<Scalar,Dim> & bullet_velocity)
{
    this->bullet_velocity_ = bullet_velocity;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setCriticalStretchVec(const std::vector<Scalar> & s_vec)
{
    this->critical_s_vec_ = s_vec;
}

template <typename Scalar, int Dim>
std::vector<unsigned int> & PDMPluginRender<Scalar, Dim>::specifiedIdxVec()
{
    return this->specified_idx_vec_;
}

template <typename Scalar, int Dim>
const std::vector<unsigned int> & PDMPluginRender<Scalar, Dim>::specifiedIdxVec()const
{
    return this->specified_idx_vec_;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::addIdxToSpecifiedIdxVec(unsigned int par_idx)
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);
    if (par_idx >= driver->numSimParticles())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->specified_idx_vec_.push_back(par_idx);
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::addIdxToSpecifiedParVec(unsigned int par_idx)
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);
    if (par_idx >= driver->numSimParticles())
    {
        std::cerr<<" Particle index out of range. \n";
        std::exit(EXIT_FAILURE);
    }
    this->specified_par_vec_.push_back(par_idx);
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::setWindow(GlutWindow * window)
{
    if(window==NULL)
    {
        std::cerr<<"Error: NULL window pointer provided to render plugin, program abort!\n";
        std::exit(EXIT_FAILURE);
    }
    this->window_ = window;
    // automatically set idle function, display function, and keyboard function
    this->window_->setIdleFunction(PDMPluginRender<Scalar,Dim>::idleFunction);
    this->window_->setDisplayFunction(PDMPluginRender<Scalar,Dim>::displayFunction);
    this->window_->setKeyboardFunction(PDMPluginRender<Scalar,Dim>::keyboardFunction);
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderParticle()
{
    this->render_particle_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderFamily()
{
    this->render_specified_family_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderVelocity()
{
    this->render_velocity_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderFloor()
{
    this->render_floor_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderSpecifiedParticle()
{
    this->render_specified_particle_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderBullet()
{
    this->render_bullet_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderCriticalStretch()
{
    this->render_s_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderCurSize()
{
    this->render_cur_size_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::enableRenderMesh()
{
    this->render_mesh_ = true;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderParticle()
{
    this->render_particle_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderFamily()
{
    this->render_specified_family_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderVelocity()
{
    this->render_velocity_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderFloor()
{
    this->render_floor_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderSpecifiedParticle()
{
    this->render_specified_particle_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderBullet()
{
    this->render_bullet_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderCriticalStretch()
{
    this->render_s_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderCurSize()
{
    this->render_cur_size_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::disableRenderMesh()
{
    this->render_mesh_ = false;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::idleFunction(void)
{

}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::displayFunction(void)
{
    PHYSIKA_ASSERT(active_instance_);
    GlutWindow *window = active_instance_->window_;
    PHYSIKA_ASSERT(window);
    Color<double> background_color = window->backgroundColor<double>();
    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    
    //window->applyCameraAndLights();

    RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
    Camera<double> & camera = render_scene_config.camera();
    camera.look();

    // render particles
    if(active_instance_->render_particle_)
        active_instance_->renderParticles();

    // render specified Idx family
    if(active_instance_->render_specified_family_)
    {
        for(unsigned int i=0; i<active_instance_->specified_idx_vec_.size(); i++)
        {
            active_instance_->renderParticleFamily(active_instance_->specified_idx_vec_[i]);
        }
    }
    // render velocity
    if(active_instance_->render_velocity_)
    {
        active_instance_->renderParticleVelocity();
    }
    
    if (active_instance_->render_floor_)
    {
        active_instance_->renderFloor();
    }

    if (active_instance_->render_compression_plane_)
    {
        active_instance_->renderCompressionPlane();
    }

    if (active_instance_->render_specified_particle_)
    {
        active_instance_->renderSpeicifiedParticles();
    }

    if (active_instance_->render_bullet_)
    {
        active_instance_->renderBullet();
    }

    if (active_instance_->render_mesh_)
    {
        active_instance_->renderMesh();
    }

    window->displayFrameRate();
    glutSwapBuffers();

}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderParticles()
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);

    unsigned int num_particles = driver->numSimParticles();
    Vector<Scalar, Dim> * particle_pos = new Vector<Scalar, Dim>[num_particles];
    for(unsigned int i = 0; i < num_particles; i++)
    {
        particle_pos[i] = driver->particleCurrentPosition(i);
    }

    PointRender<Scalar, Dim> point_render(particle_pos, num_particles);
    point_render.setPointSize(this->point_size_);

    std::vector<Color<Scalar> > color_vec;
    if (driver->isSimulationPause() == false)
    {
        if (this->render_cur_size_ == false)
        {
            for (unsigned int i=0; i< num_particles; i++)
            {
                const PDMParticle<Scalar, Dim> & particle = driver->particle(i);
                Scalar demage_ratio = 1.0 - static_cast<Scalar>(particle.validFamilySize())/particle.initFamilySize();
                const Color<Scalar> & color = color_map_.colorViaRatio(demage_ratio);
                color_vec.push_back(color);
            }
        }
        else
        {
            for (unsigned int i=0; i< num_particles; i++)
            {
                const PDMParticle<Scalar, Dim> & particle = driver->particle(i);
                Scalar demage_ratio = static_cast<Scalar>(particle.validFamilySize())/particle.initFamilySize();
                const Color<Scalar> & color = color_map_.colorViaRatio(demage_ratio);
                color_vec.push_back(color);
            }
        }
        
    }
    else if (this->render_s_)
    {
        for (unsigned int i=0; i<num_particles; i++)
        {
            if (critical_s_vec_[i]<=0.0001)
            {
                const Color<Scalar> & color = color_map_.colorViaRatio(1.0);
                color_vec.push_back(color);
            }
            else
            {
                const Color<Scalar> & color = color_map_.colorViaRatio(0.0);
                color_vec.push_back(color);
            }
        }
    }
    else
    {
        color_vec.insert(color_vec.begin(),num_particles,Color<Scalar>(0,1.0,0));
    }
    
    point_render.setPointColor(color_vec);

    //point_render.setRenderAsSphere();
    point_render.render();
    // free memory
    delete [] particle_pos;
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderParticleVelocity()
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);
    openGLColor3(Color<Scalar>::Red());
    glPushAttrib(GL_COLOR_BUFFER_BIT|GL_CURRENT_BIT);
    glDisable(GL_LIGHTING);

    unsigned int num_particles = driver->numSimParticles();
    for (unsigned int i=0; i<num_particles; i++)
    {
        Vector<Scalar, Dim> start = driver->particleCurrentPosition(i);
        Vector<Scalar, Dim> end = start + (this->velocity_scale_)*driver->particleVelocity(i);
        glBegin(GL_LINES);
        openGLVertex(start);
        openGLVertex(end);
        glEnd();
    }
    glPopAttrib();
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderParticleFamily(unsigned int par_idx)
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);

    // render particle family
    PDMParticle<Scalar, Dim> & particle_idx = driver->particle(par_idx);
    Vector<Scalar, Dim> * particle_family_pos = new Vector<Scalar, Dim> [particle_idx.family().size()];
    unsigned int i=0;
    for (typename std::list<PDMFamily<Scalar, Dim>>::iterator iter = particle_idx.family().begin(); iter != particle_idx.family().end(); iter++,i++)
    {
        particle_family_pos[i] = driver->particleCurrentPosition((*iter).id());
    }
    PointRender<Scalar, Dim> point_render(particle_family_pos, particle_idx.family().size());
    point_render.setPointColor(Color<Scalar>::Red());
    point_render.setPointSize(1.5*this->point_size_);
    point_render.render();
    delete [] particle_family_pos;

    // render particle itself
    Vector<Scalar, Dim> particle_pos = driver->particleCurrentPosition(par_idx);
    point_render.setPoints(&particle_pos,1);
    point_render.setPointSize(3.0*this->point_size_);
    point_render.setPointColor(Color<Scalar>::Blue());
    point_render.render();
    
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderFloor()
{
    openGLColor3(Color<Scalar>(0.45,0.45,0.45));
    Vector<Scalar, 3> ver_1(-20,this->floor_pos_, 20);
    Vector<Scalar, 3> ver_2(20 ,this->floor_pos_, 20);
    Vector<Scalar, 3> ver_3(20 ,this->floor_pos_,-20);
    Vector<Scalar, 3> ver_4(-20,this->floor_pos_,-20);
    glBegin(GL_QUADS);
        openGLVertex(ver_1);
        openGLVertex(ver_2);
        openGLVertex(ver_3);
        openGLVertex(ver_4);
    glEnd();
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderCompressionPlane()
{
    openGLColor3(Color<Scalar>(0.45,0.45,0.45));

    Vector<Scalar, 3> up_ver_1(-3,this->up_plane_pos_, 3);
    Vector<Scalar, 3> up_ver_2(3 ,this->up_plane_pos_, 3);
    Vector<Scalar, 3> up_ver_3(3 ,this->up_plane_pos_,-3);
    Vector<Scalar, 3> up_ver_4(-3,this->up_plane_pos_,-3);
    glBegin(GL_QUADS);
    openGLVertex(up_ver_1);
    openGLVertex(up_ver_2);
    openGLVertex(up_ver_3);
    openGLVertex(up_ver_4);
    glEnd();

    Vector<Scalar, 3> down_ver_1(-3,this->down_plane_pos_, 3);
    Vector<Scalar, 3> down_ver_2(3 ,this->down_plane_pos_, 3);
    Vector<Scalar, 3> down_ver_3(3 ,this->down_plane_pos_,-3);
    Vector<Scalar, 3> down_ver_4(-3,this->down_plane_pos_,-3);
    glBegin(GL_QUADS);
    openGLVertex(down_ver_1);
    openGLVertex(down_ver_2);
    openGLVertex(down_ver_3);
    openGLVertex(down_ver_4);
    glEnd();
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderSpeicifiedParticles()
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);

    for(unsigned int i=0; i<active_instance_->specified_par_vec_.size(); i++)
    {
        Vector<Scalar,Dim> position = driver->particleCurrentPosition(specified_par_vec_[i]);

        PointRender<Scalar, Dim> point_render(&position, 1);
        point_render.setPointSize(this->point_size_*2);
        point_render.setPointColor(Color<Scalar>::Red());
        point_render.render();
    }  
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderBullet()
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);

    Vector<Scalar, Dim> bullet_pos = this->bullet_pos_;
    Scalar bullet_radius = this->bullet_radius_;

    //if impact method is speicified, use its info
    if (this->impact_method_ != NULL)
    {
        bullet_pos = this->impact_method_->impactPos();
        bullet_radius = this->impact_method_->impactRadius();
    }

    glEnable(GL_LIGHTING);
    glPushMatrix();
        openGLTranslate(bullet_pos);
        glutSolidSphere(bullet_radius, 100, 100);
    glPopMatrix();
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::renderMesh()
{
    PDMBase<Scalar, Dim> * driver = this->driver();
    PHYSIKA_ASSERT(driver);

    VolumetricMeshRender<Scalar, Dim> mesh_render;
    mesh_render.setVolumetricMesh(driver->mesh());
    mesh_render.disableRenderSolid();
    mesh_render.enableRenderWireframe();
    mesh_render.render();
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::keyboardFunction(unsigned char key, int x, int y)
{
    PHYSIKA_ASSERT(active_instance_);
    GlutWindow * window = active_instance_->window_;
    PHYSIKA_ASSERT(window);
    // default key is preserved
    window->bindDefaultKeys(key, x, y);
    switch(key)
    {
    case 32:
        {
            bool pause_simulation = active_instance_->driver()->isSimulationPause();
            
            if (pause_simulation)
            {
                active_instance_->driver()->forwardSimulation();
            }
            else
            {
                active_instance_->driver()->pauseSimulation();
            }
            break;
        }
    case 'S':
        active_instance_->auto_capture_frame_ = !(active_instance_->auto_capture_frame_);
        break;
    case 'm':
        active_instance_->render_mesh_ = !active_instance_->render_mesh_;
        break;
    case 'c':
        {
            PDMStepMethodBase<Scalar, Dim> * step_method = active_instance_->driver()->stepMethod();
            if (step_method->isCollisionEnabled())
                step_method->disableCollision();
            else
                step_method->enableCollision();
            break;
        }
    case 'p':
        {
            PDMStepMethodStateViscoPlasticity<Scalar, Dim> * step_method = dynamic_cast<PDMStepMethodStateViscoPlasticity<Scalar, Dim> *> (active_instance_->driver()->stepMethod());
            if (step_method->isPlasticStatisticsEnabled())
                step_method->disablePlasticStatistics();
            else
                step_method->enablePlasticStatistics();
            break;
        }
    default:
        break;
    }
}

template <typename Scalar, int Dim>
void PDMPluginRender<Scalar, Dim>::activateCurrentInstance()
{
    this->active_instance_ = this;
}

// explicit instantiations
template class PDMPluginRender<float, 2>;
template class PDMPluginRender<float, 3>;
template class PDMPluginRender<double, 2>;
template class PDMPluginRender<double, 3>;

}