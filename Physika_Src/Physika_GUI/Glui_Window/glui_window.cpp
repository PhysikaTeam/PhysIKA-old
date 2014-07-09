/*
 * @file glui_window.cpp
 * @Brief Glui-based window, subclass of GlutWindow, supports GLUI controls.
 *        Provides all features of GlutWindow, except that closing the window will terminate the program.
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
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_GUI/Glui_Window/glui_window.h"

namespace Physika{

int GluiWindow::main_window_id_ = -1;

GluiWindow::GluiWindow()
    :GlutWindow()
{
    initGLUT();
    idle_function_ = GluiWindow::idleFunction;
    glui_ =GLUI_Master.create_glui("Control Panel");
}

GluiWindow::GluiWindow(const std::string &window_name)
    :GlutWindow(window_name)
{
    initGLUT();
    idle_function_ = GluiWindow::idleFunction;
    glui_ =GLUI_Master.create_glui("Control Panel");
}

GluiWindow::GluiWindow(const std::string &window_name, unsigned int width, unsigned int height)
        :GlutWindow(window_name,width,height)
{
    initGLUT();
    idle_function_ = GluiWindow::idleFunction;
    glui_ =GLUI_Master.create_glui("Control Panel");
}

GluiWindow::~GluiWindow()
{
    GLUI_Master.close_all();
}

void GluiWindow::createWindow()
{
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH|GLUT_ALPHA);
    glutInitWindowSize(initial_width_,initial_height_);
    window_id_ = glutCreateWindow(window_name_.c_str());
    GluiWindow::main_window_id_ = window_id_;  //update current main window id, for use in idleFunction
    glutSetWindowData(this);  //bind 'this' pointer with the window
    glutDisplayFunc(display_function_);
    glutReshapeFunc(reshape_function_);
    glutKeyboardFunc(keyboard_function_);
    glutSpecialFunc(special_function_);
    glutMotionFunc(motion_function_);
    glutMouseFunc(mouse_function_);
    PHYSIKA_ASSERT(glui_);
    glui_->set_main_gfx_window(window_id_);
    GLUI_Master.set_glutIdleFunc(idle_function_);
    PHYSIKA_ASSERT(init_function_);
    (*init_function_)(); //call the init function before entering main loop
    glutMainLoop();
}

GLUI* GluiWindow::gluiWindow()
{
    return glui_;
}

int GluiWindow::mainWindowId()
{
    return main_window_id_;
}

void GluiWindow::initGLUT()
{
    //init GLUT
    int argc = 1;
    const int max_length = 1024; //assume length of the window name does not exceed 1024 characters
    char *argv[1];
    char name_str[max_length];
    strcpy(name_str,window_name_.c_str());
    argv[0] = name_str;
    glutInit(&argc,argv);
}

void GluiWindow::idleFunction(void)
{
    glutSetWindow(GluiWindow::main_window_id_); //set the current GLUT window
    glutPostRedisplay();
}

}  //end of namespace Physika
