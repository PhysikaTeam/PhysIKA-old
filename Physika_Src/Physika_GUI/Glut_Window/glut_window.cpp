/*
 * @file glut_window.cpp 
 * @Brief Glut-based window, provide default response functions and support custom response functions.
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

#include <cstring>
#include <iostream>
#include <GL/freeglut.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Render/Color/color.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

namespace Physika{

GlutWindow::GlutWindow()
    :window_name_(std::string("Physika Glut Window")),window_id_(-1),initial_width_(640),initial_height_(480)
{
    camera_.setCameraAspect((GLdouble)initial_width_/initial_height_);
    initCallbacks();
}

GlutWindow::GlutWindow(const std::string &window_name)
    :window_name_(window_name),window_id_(-1),initial_width_(640),initial_height_(480)
{
    camera_.setCameraAspect((GLdouble)initial_width_/initial_height_);
    initCallbacks();
}

GlutWindow::GlutWindow(const std::string &window_name, unsigned int width, unsigned int height)
    :window_name_(window_name),window_id_(-1),initial_width_(width),initial_height_(height)
{
    camera_.setCameraAspect((GLdouble)initial_width_/initial_height_);
    initCallbacks();
}

GlutWindow::~GlutWindow()
{
}

void GlutWindow::createWindow()
{
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

void GlutWindow::translateCamera(const Vector<double,3> &vec)
{
    camera_.translate(vec);
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

const RenderBase* GlutWindow::renderTaskAtIndex(unsigned int index) const
{
    return render_manager_.taskAtIndex(index);
}

RenderBase* GlutWindow::renderTaskAtIndex(unsigned int index)
{
    return render_manager_.taskAtIndex(index);
}

int GlutWindow::renderTaskIndex(RenderBase *task) const
{
    return render_manager_.taskIndex(task);
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

////////////////////////////////////////////////// default callback functions ////////////////////////////////////////////////////////////////////

void GlutWindow::displayFunction(void)
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glutSwapBuffers();
}

void GlutWindow::idleFunction(void)
{
    glutPostRedisplay();
}

void GlutWindow::reshapeFunction(int width, int height)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    glViewport(0,0,width,height);
    window->setCameraAspect((GLdouble)width/height); //update the  view aspect of camera
    (window->camera_).look(); //apply the new camera
}

void GlutWindow::keyboardFunction(unsigned char key, int x, int y)
{
    switch(key)
    {
    case 27: //ESC: close window
        glutLeaveMainLoop();
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
}

void GlutWindow::mouseFunction(int button, int state, int x, int y)
{
}

void GlutWindow::initFunction(void)
{
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    glMatrixMode(GL_PROJECTION);												// select projection matrix
    glViewport(0, 0, width, height);        									// set the viewport
    glMatrixMode(GL_PROJECTION);												// set matrix mode
    glLoadIdentity();															// reset projection matrix
    gluPerspective(45.0,(GLdouble)width/height,1.0e-3,1.0e2);           		// set up a perspective projection matrix
    glMatrixMode(GL_MODELVIEW);													// specify which matrix is the current matrix
    glShadeModel( GL_SMOOTH );
    glClearDepth( 1.0 );														// specify the clear value for the depth buffer
    glEnable( GL_DEPTH_TEST );
    glDepthFunc( GL_LEQUAL );
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );						// specify implementation-specific hints
    Color<unsigned char> black = Color<unsigned char>::Black();
    glClearColor(black.redChannel(), black.greenChannel(), black.blueChannel(), black.alphaChannel());	
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

} //end of namespace Physika
