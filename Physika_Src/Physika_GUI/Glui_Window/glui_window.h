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

#ifndef PHYSIKA_GUI_GLUI_WINDOW_GLUI_WINDOW_H_
#define PHYSIKA_GUI_GLUI_WINDOW_GLUI_WINDOW_H_

#include <string>
#include "Physika_GUI/Glut_Window/glut_window.h"
class GLUI;

namespace Physika{

class GluiWindow: public GlutWindow
{
public:
    GluiWindow();  //initialize a window with default name and size
    GluiWindow(const std::string &window_name); //initialize a window with given name and default size
    GluiWindow(const std::string &window_name, unsigned int width, unsigned int height); //initialize a window with given name and size
    ~GluiWindow();
    void createWindow(GLUI *glui);  //create window with GLUI control, create a GlutWindow if NULL pointer passed
    void closeWindow();  //close window
protected:
    void initGLUT();  //init GLUT stuff
};

}  //end of namespace Physika

#endif //PHYSIKA_GUI_GLUI_WINDOW_GLUI_WINDOW_H_
