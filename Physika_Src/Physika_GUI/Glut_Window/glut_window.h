/*
 * @file glut_window.h 
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

#ifndef PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_
#define PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_

#include <string>

namespace Physika{

class GlutWindow
{
public:
    GlutWindow();  //initialize a window with default name and size
    GlutWindow(const std::string &window_name, unsigned int width, unsigned int height);
    ~GlutWindow();
protected:
    std::string window_name_;
    unsigned int width_;
    unsigned int height_;
};

}  //end of namespace Physika

#endif  //PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_
