/*
 * @file glui_window.h
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

#ifndef PHYSIKA_GUI_GLUI_WINDOW_GLUI_WINDOW_H_
#define PHYSIKA_GUI_GLUI_WINDOW_GLUI_WINDOW_H_

#include <string>
#include "Physika_GUI/Glut_Window/glut_window.h"
class GLUI;

namespace Physika{

/*
 * Glui-based window
 * Key features:
 *       All features of GlutWindows except that closing the window will terminate the program
 * Usage:
 *       1. Define a GluiWindow object
 *       2. Set the custom callback functions (optional)
 *       3. Call gluiWindow() to get the pointer of the GLUI pannel
 *       4. Add standard GLUI controls to the GLUI pannel
 *       5. Call createWindow() 
 *       6. Call closeWindow() or click the 'X' on window to close the window
 * Note: 
 *      GLUI requires specially handling in idleFunction, so be sure to add following code at the
 *      begining of your custom idle function so that the glut window is the main part instead of
 *      the control panel:
 *      
 *      glutSetWindow(GluiWindow::mainWindowId());
 *
 *      GluiWindow::mainWindowId() returns the id of the main window of currently created GluiWindow
 *      At one time only one GluiWindow  window can be created
 */

class GluiWindow: public GlutWindow
{
public:
    GluiWindow();  //initialize a window with default name and size
    GluiWindow(const std::string &window_name); //initialize a window with given name and default size
    GluiWindow(const std::string &window_name, unsigned int width, unsigned int height); //initialize a window with given name and size
    ~GluiWindow();
    void createWindow();  //create window with GLUI control
    GLUI* gluiWindow(); //return pointer to the GLUI control window
    static int mainWindowId(); //get the GLUT window id of currently opened GluiWindow
protected:
    void initGLUT();  //init GLUT
    static void idleFunction(void); //overwritten the default GlutWindow idle function
protected:
    GLUI *glui_;
    static int main_window_id_;
};

}  //end of namespace Physika

#endif //PHYSIKA_GUI_GLUI_WINDOW_GLUI_WINDOW_H_
