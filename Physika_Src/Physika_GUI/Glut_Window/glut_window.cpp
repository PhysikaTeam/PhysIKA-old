/*
 * @file glut_window.cpp 
 * @Brief Glut-based window.
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
#include <cstring>
#include <iostream>
#include <sstream>
#include <GL/freeglut.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/OpenGL_Primitives/opengl_primitives.h"
#include "Physika_Core/Image/image.h"
#include "Physika_IO/Image_IO/image_io.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

namespace Physika{

GlutWindow::GlutWindow()
    :window_name_(std::string("Physika Glut Window")),window_id_(-1),initial_width_(640),initial_height_(480),
     display_fps_(true),screen_capture_file_index_(0)
{
    background_color_ = Color<double>::Black();
    text_color_ = Color<double>::White();
    resetMouseState();
    camera_.setCameraAspect((GLdouble)initial_width_/initial_height_);
    initCallbacks();
    light_manager_.insertBack(&default_light_);
}

GlutWindow::GlutWindow(const std::string &window_name)
    :window_name_(window_name),window_id_(-1),initial_width_(640),initial_height_(480),
     display_fps_(true),screen_capture_file_index_(0)
{
    background_color_ = Color<double>::Black();
    text_color_ = Color<double>::White();
    resetMouseState();
    camera_.setCameraAspect((GLdouble)initial_width_/initial_height_);
    initCallbacks();
    light_manager_.insertBack(&default_light_);
}

GlutWindow::GlutWindow(const std::string &window_name, unsigned int width, unsigned int height)
    :window_name_(window_name),window_id_(-1),initial_width_(width),initial_height_(height),
     display_fps_(true),screen_capture_file_index_(0)
{
    background_color_ = Color<double>::Black();
    text_color_ = Color<double>::White();
    resetMouseState();
    camera_.setCameraAspect((GLdouble)initial_width_/initial_height_);
    initCallbacks();
    default_light_.turnOn();
    light_manager_.insertBack(&default_light_);
}

GlutWindow::~GlutWindow()
{
}

void GlutWindow::createWindow()
{
    resetMouseState(); //reset the state of mouse every time the window is created
    int argc = 1;
    const int max_length = 1024; //assume length of the window name does not exceed 1024 characters
    char *argv[1];
    char name_str[max_length];
    strcpy(name_str,window_name_.c_str());
    argv[0] = name_str;
    glutInit(&argc,argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);  //this option allows leaving the glut loop without exit program
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH|GLUT_ALPHA);
    glutInitWindowSize(initial_width_,initial_height_);
    window_id_ = glutCreateWindow(name_str);
    glutSetWindowData(this);  //bind 'this' pointer with the window
    glutDisplayFunc(display_function_);
    glutIdleFunc(idle_function_);
    glutReshapeFunc(reshape_function_);
    glutKeyboardFunc(keyboard_function_);
    glutSpecialFunc(special_function_);
    glutMotionFunc(motion_function_);
    glutMouseFunc(mouse_function_);
    (*init_function_)(); //call the init function before entering main loop
    glutMainLoop();
}

void GlutWindow::closeWindow()
{
    glutLeaveMainLoop();
}

const std::string& GlutWindow::name() const
{
    return window_name_;
}

int GlutWindow::width() const
{
    if(glutGet(GLUT_INIT_STATE))  //window is created
        return glutGet(GLUT_WINDOW_WIDTH);
    else
        return initial_width_;
}

int GlutWindow::height() const
{
    if(glutGet(GLUT_INIT_STATE)) //window is created
        return glutGet(GLUT_WINDOW_HEIGHT);
    else
        return initial_height_;
}

////////////////////////////////////////////////// camera operations////////////////////////////////////////////////////////////////////////

const Vector<double,3>& GlutWindow::cameraPosition() const
{
    return camera_.cameraPosition();
}

void GlutWindow::setCameraPosition(const Vector<double,3> &position)
{
    camera_.setCameraPosition(position);
}

const Vector<double,3>& GlutWindow::cameraUpDirection() const
{
    return camera_.cameraUpDirection();
}

void GlutWindow::setCameraUpDirection(const Vector<double,3> &up)
{
    camera_.setCameraUpDirection(up);
}

const Vector<double,3>& GlutWindow::cameraFocusPosition() const
{
    return camera_.cameraFocusPosition();
}

void GlutWindow::setCameraFocusPosition(const Vector<double,3> &focus)
{
    camera_.setCameraFocusPosition(focus);
}

double GlutWindow::cameraFOV() const
{
    return camera_.cameraFOV();
}

void GlutWindow::setCameraFOV(double fov)
{
    camera_.setCameraFOV(fov);
}

double GlutWindow::cameraAspect() const
{
    return camera_.cameraAspect();
}

void GlutWindow::setCameraAspect(double aspect)
{
    camera_.setCameraAspect(aspect);
}

double GlutWindow::cameraNearClip() const
{
    return camera_.cameraNearClip();
}

void GlutWindow::setCameraNearClip(double near_clip)
{
    camera_.setCameraNearClip(near_clip);
}

double GlutWindow::cameraFarClip() const
{
    return camera_.cameraFarClip();
}

void GlutWindow::setCameraFarClip(double far_clip)
{
    camera_.setCameraFarClip(far_clip);
}

void GlutWindow::orbitCameraUp(double rad)
{
    camera_.orbitUp(rad);
}

void GlutWindow::orbitCameraDown(double rad)
{
    camera_.orbitDown(rad);
}

void GlutWindow::orbitCameraLeft(double rad)
{
    camera_.orbitLeft(rad);
}

void GlutWindow::orbitCameraRight(double rad)
{
    camera_.orbitRight(rad);
}

void GlutWindow::zoomCameraIn(double dist)
{
    camera_.zoomIn(dist);
}

void GlutWindow::zoomCameraOut(double dist)
{
    camera_.zoomOut(dist);
}

void GlutWindow::yawCamera(double rad)
{
    camera_.yaw(rad);
}

void GlutWindow::pitchCamera(double rad)
{
    camera_.pitch(rad);
}

void GlutWindow::rollCamera(double rad)
{
    camera_.roll(rad);
}

void GlutWindow::translateCameraUp(double dist)
{
    camera_.translateUp(dist);
}

void GlutWindow::translateCameraDown(double dist)
{
    camera_.translateDown(dist);
}

void GlutWindow::translateCameraLeft(double dist)
{
    camera_.translateLeft(dist);
}

void GlutWindow::translateCameraRight(double dist)
{
    camera_.translateRight(dist);
}

////////////////////////////////////////////////// manages lights in scene///////////////////////////////////////////////////////////////////////////

unsigned int GlutWindow::numLights() const
{
    return light_manager_.numLights();
}

void GlutWindow::pushBackLight(Light *light)
{
    light_manager_.insertBack(light);
}

void GlutWindow::pushFrontLight(Light *light)
{
    light_manager_.insertFront(light);
}

void GlutWindow::insertLightAtIndex(unsigned int index, Light *light)
{
    light_manager_.insertAtIndex(index,light);
}

void GlutWindow::popBackLight()
{
    light_manager_.removeBack();
}

void GlutWindow::popFrontLight()
{
    light_manager_.removeFront();
}

void GlutWindow::removeLightAtIndex(unsigned int index)
{
    light_manager_.removeAtIndex(index);
}

void GlutWindow::removeAllLights()
{
    light_manager_.removeAll();
}

const Light* GlutWindow::lightAtIndex(unsigned int index) const
{
    return light_manager_.lightAtIndex(index);
}

Light* GlutWindow::lightAtIndex(unsigned int index)
{
    return light_manager_.lightAtIndex(index);
}

int GlutWindow::lightIndex(Light *light) const
{
    return light_manager_.lightIndex(light);
}

////////////////////////////////////////////////// manages render tasks////////////////////////////////////////////////////////////////////

unsigned int GlutWindow::numRenderTasks() const
{
    return render_manager_.numRenderTasks();
}

void GlutWindow::pushBackRenderTask(RenderBase *task)
{
    render_manager_.insertBack(task);
}

void GlutWindow::pushFrontRenderTask(RenderBase *task)
{
    render_manager_.insertFront(task);
}

void GlutWindow::insertRenderTaskAtIndex(unsigned int index, RenderBase *task)
{
    render_manager_.insertAtIndex(index,task);
}

void GlutWindow::popBackRenderTask()
{
    render_manager_.removeBack();
}

void GlutWindow::popFrontRenderTask()
{
    render_manager_.removeFront();
}

void GlutWindow::removeRenderTaskAtIndex(unsigned int index)
{
    render_manager_.removeAtIndex(index);
}

void GlutWindow::removeAllRenderTasks()
{
    render_manager_.removeAll();
}

const RenderBase* GlutWindow::getRenderTaskAtIndex(unsigned int index) const
{
    return render_manager_.taskAtIndex(index);
}

RenderBase* GlutWindow::getRenderTaskAtIndex(unsigned int index)
{
    return render_manager_.taskAtIndex(index);
}

int GlutWindow::getRenderTaskIndex(RenderBase *task) const
{
    return render_manager_.taskIndex(task);
}

////////////////////////////////////////////////// screen shot and display frame-rate////////////////////////////////////////////////////////////////

bool GlutWindow::saveScreen(const std::string &file_name) const
{
    int width = this->width(), height = this->height();
    unsigned char *data = new unsigned char[width*height*3];  //RGB
    PHYSIKA_ASSERT(data);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0,0,width,height,GL_RGB,GL_UNSIGNED_BYTE,(void*)data);
    Image image(width,height,Image::RGB,data);
    image.flipVertically();
    bool status = ImageIO::save(file_name,&image);
    delete[] data;
	return status;
}

bool GlutWindow::saveScreen()
{
    std::stringstream adaptor;
    adaptor<<screen_capture_file_index_++;
    std::string index_str;
    adaptor>>index_str;
    std::string file_name = std::string("screen_capture_") + index_str + std::string(".png"); 
    return saveScreen(file_name);
}

void GlutWindow::displayFrameRate() const
{
    if(!glutGet(GLUT_INIT_STATE))  //window is not created
    {
        std::cerr<<"Cannot display frame rate before a window is created.\n";
        std::exit(EXIT_FAILURE);
    }
    if(display_fps_)
    {
        static unsigned int frame = 0, time = 0, time_base = 0;
        double fps = 60.0;
        ++frame;
        time = glutGet(GLUT_ELAPSED_TIME); //millisecond
        if(time - time_base > 10) // compute every 10 milliseconds
        {
            fps = frame*1000.0/(time-time_base);
            time_base = time;
            frame = 0;
        }
        std::stringstream adaptor;
        adaptor.precision(2);
        std::string str;
        if(fps>1.0)  //show fps
        {
            adaptor<<fps;
            str = std::string("FPS: ") + adaptor.str();
        }
        else  //show spf
        {
            PHYSIKA_ASSERT(fps>0);
            adaptor<< 1.0/fps;
            str = std::string("SPF: ") + adaptor.str();
        }
        //now draw the string
        int width = this->width(), height = this->height();
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadIdentity();
        glDisable(GL_LIGHTING);
        glDisable(GL_TEXTURE_2D);
        gluOrtho2D(0,width,0,height);
        openGLColor3(text_color_);
        glRasterPos2i(5,height-19);

        for (unsigned int i = 0; i < str.length(); i++) 
            glutBitmapCharacter (GLUT_BITMAP_HELVETICA_18, str[i]);

        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        glPopMatrix();
        glEnable(GL_LIGHTING);
    }
}

void GlutWindow::enableDisplayFrameRate()
{
    display_fps_ = true;
}

void GlutWindow::disableDisplayFrameRate()
{
    display_fps_ = false;
}

void GlutWindow::applyCameraAndLights()
{
    camera_.look();
    //light the scene with the lights after calling camera look()
    //such that model view matrix are correctly set
    light_manager_.lightScene();
}

////////////////////////////////////////////////// set custom callback functions ////////////////////////////////////////////////////////////////////

void GlutWindow::setDisplayFunction(void (*func)(void))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        display_function_ = GlutWindow::displayFunction;
    }
    else
        display_function_ = func;
}

void GlutWindow::setIdleFunction(void (*func)(void))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        idle_function_ = GlutWindow::idleFunction;
    }
    else
        idle_function_ = func;
}

void GlutWindow::setReshapeFunction(void (*func)(int width, int height))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        reshape_function_ = GlutWindow::reshapeFunction;
    }
    else
        reshape_function_ = func;
}

void GlutWindow::setKeyboardFunction(void (*func)(unsigned char key, int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        keyboard_function_ = GlutWindow::keyboardFunction;
    }
    else
        keyboard_function_ = func;
}

void GlutWindow::setSpecialFunction(void (*func)(int key, int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        special_function_ = GlutWindow::specialFunction;
    }
    else
        special_function_ = func;
}

void GlutWindow::setMotionFunction(void (*func)(int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        motion_function_ = GlutWindow::motionFunction;
    }
    else
        motion_function_ = func;
}

void GlutWindow::setMouseFunction(void (*func)(int button, int state, int x, int y))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        mouse_function_ = GlutWindow::mouseFunction;
    }
    else
        mouse_function_ = func;
}

void GlutWindow::setInitFunction(void (*func)(void))
{
    if(func==NULL)
    {
        std::cerr<<"NULL callback function provided, use default instead.\n";
        init_function_ = GlutWindow::initFunction;
    }
    else
        init_function_ = func;
}

void GlutWindow::bindDefaultKeys(unsigned char key, int x, int y)
{
    GlutWindow::keyboardFunction(key,x,y);
}

////////////////////////////////////////////////// default callback functions ////////////////////////////////////////////////////////////////////

void GlutWindow::displayFunction(void)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);
    Color<double> background_color = window->background_color_;
    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    window->applyCameraAndLights();
    (window->render_manager_).renderAll(); //render all tasks of render manager
    window->displayFrameRate();
    glutSwapBuffers();
}

void GlutWindow::idleFunction(void)
{
    glutPostRedisplay();
}

void GlutWindow::reshapeFunction(int width, int height)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);
    GLdouble aspect = static_cast<GLdouble>(width)/height;
    window->setCameraAspect(aspect); //update the  view aspect of camera
    double fov = window->cameraFOV();
    double near_clip = window->cameraNearClip();
    double far_clip = window->cameraFarClip();
    //update view port and projection
    glViewport(0,0,width,height);
    glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, aspect,near_clip,far_clip);
	glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void GlutWindow::keyboardFunction(unsigned char key, int x, int y)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);
    switch(key)
    {
    case 27: //ESC: close window
        glutLeaveMainLoop();
        break;
    case 's': //s: save screen shot
        window->saveScreen();
        break;
    case 'f': //f: enable/disable FPS display
        (window->display_fps_) = !(window->display_fps_);
        break;
    default:
        break;
    }
}

void GlutWindow::specialFunction(int key, int x, int y)
{
}

void GlutWindow::motionFunction(int x, int y)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);
    int mouse_delta_x = x - window->mouse_position_[0];
    int mouse_delta_y = y - window->mouse_position_[1];
    window->mouse_position_[0] = x;
    window->mouse_position_[1] = y;
    double scale = 0.02;  //sensativity of the mouse
    double camera_radius = (window->cameraFocusPosition()-window->cameraPosition()).norm();
    if(window->left_button_down_)  //left button handles camera rotation
    {
        window->orbitCameraLeft(mouse_delta_x*scale);
        window->orbitCameraUp(mouse_delta_y*scale);
    }
    if(window->middle_button_down_)  //middle button handles camera zoom in/out
    {
        window->zoomCameraIn(camera_radius*mouse_delta_y*scale);
    }
    if(window->right_button_down_)  //right button handles camera translation
    {
        scale *= 0.1;
        window->translateCameraLeft(camera_radius*mouse_delta_x*scale);
        window->translateCameraUp(camera_radius*mouse_delta_y*scale);
    }
}

void GlutWindow::mouseFunction(int button, int state, int x, int y)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);
    switch(button)
    {
    case GLUT_LEFT_BUTTON:
        window->left_button_down_ = (state==GLUT_DOWN);
        break;
    case GLUT_MIDDLE_BUTTON:
        window->middle_button_down_ = (state==GLUT_DOWN);
        break;
    case GLUT_RIGHT_BUTTON:
        window->right_button_down_ = (state==GLUT_DOWN);
        break;
    default:
        //PHYSIKA_ERROR("Invalid mouse state.");
        break;
    }
    window->mouse_position_[0] = x;
    window->mouse_position_[1] = y;
}

void GlutWindow::initFunction(void)
{
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);
    glMatrixMode(GL_PROJECTION);												// select projection matrix
    glViewport(0, 0, width, height);        									// set the viewport
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    window->initDefaultLight();
    window->applyCameraAndLights();                                             // apply the camera and lights
    glShadeModel( GL_SMOOTH );
    glClearDepth( 1.0 );														// specify the clear value for the depth buffer
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );						// specify implementation-specific hints
    Color<double> background_color = window->background_color_;
    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
}

void GlutWindow::initCallbacks()
{
    //set callbacks to default callback functions
    display_function_ = GlutWindow::displayFunction;
    idle_function_ = GlutWindow::idleFunction;
    reshape_function_ = GlutWindow::reshapeFunction;
    keyboard_function_ = GlutWindow::keyboardFunction;
    special_function_ = GlutWindow::specialFunction;
    motion_function_ = GlutWindow::motionFunction;
    mouse_function_ = GlutWindow::mouseFunction;
    init_function_ = GlutWindow::initFunction;
}

void GlutWindow::resetMouseState()
{
    left_button_down_ = false;
    middle_button_down_ = false;
    right_button_down_ = false;
    mouse_position_[0] = mouse_position_[1] = 0;
}

void GlutWindow::initDefaultLight()
{
    default_light_.setPosition(Vector<float,3>(500, 500, 500));
    default_light_.setAmbient(Color<float>(0,0,0,1));
    default_light_.setDiffuse(Color<float>(1,1,1,1));
    default_light_.setSpecular(Color<float>(1,1,1,1));
    default_light_.turnOn();
}

} //end of namespace Physika
