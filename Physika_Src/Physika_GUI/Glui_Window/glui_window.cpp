/*
 * @file glui_window.cpp
 * @Brief Glui-based window, subclass of GluiWindow, provides all features of GlutWindow,
 *        and supports GLUI controls.
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

#include <iostream>
#include <GL/freeglut.h>
#include <GL/glui.h>
#include "Physika_GUI/Glui_Window/glui_window.h"

namespace Physika{

GluiWindow::GluiWindow():GlutWindow()
{
    initGLUT();
}

GluiWindow::GluiWindow(const std::string &window_name):GlutWindow(window_name)
{
    initGLUT();
}

GluiWindow::GluiWindow(const std::string &window_name, unsigned int width, unsigned int height)
        :GlutWindow(window_name,width,height)
{
    initGLUT();
}

GluiWindow::~GluiWindow()
{
}

void GluiWindow::createWindow(GLUI *glui)
{
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE,GLUT_ACTION_CONTINUE_EXECUTION);  //this option allows leaving the glut loop without exit program
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH|GLUT_ALPHA);
    glutInitWindowSize(initial_width_,initial_height_);
    window_id_ = glutCreateWindow(window_name_.c_str());
    glutSetWindowData(this);  //bind 'this' pointer with the window
    glutDisplayFunc(display_function_);
    glutReshapeFunc(reshape_function_);
    glutKeyboardFunc(keyboard_function_);
    glutSpecialFunc(special_function_);
    glutMotionFunc(motion_function_);
    glutMouseFunc(mouse_function_);
    (*init_function_)(); //call the init function before entering main loop
    //different ways to set idle function with/without glui
    if(glui==NULL)
    {
        std::cerr<<"NULL GLUI pointer passed, create a window with no control.\n";
        glutIdleFunc(idle_function_);
        glutMainLoop();
    }
    else
    {
        glui->set_main_gfx_window(window_id_);
        GLUI_Master.set_glutIdleFunc(idle_function_);
        glutMainLoop();
    }
}

void GluiWindow::closeWindow()
{
    glutLeaveMainLoop();
}

void GluiWindow::initGLUT()
{
    int argc = 1;
    const int max_length = 1024; //assume length of the window name does not exceed 1024 characters
    char *argv[1];
    char name_str[max_length];
    strcpy(name_str,window_name_.c_str());
    argv[0] = name_str;
    glutInit(&argc,argv);
}

}  //end of namespace Physika
