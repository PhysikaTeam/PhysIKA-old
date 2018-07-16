/*
 * @file glut_window.h 
 * @Brief Glut-based window.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013- Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#ifndef PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_
#define PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_

//#include <GL/gl.h>
#include "Physika_Render/Color/color.h"
#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_GUI/BaseWindow.h"

namespace Physika{

/*
 * Glut-based window
 * Basic features:
 *     1. closing the window will not terminate the program
 *     2. provide default callback functions (see the comments of default functions to view their functionality)
 *     3. provide camera set up
 *     4. allow user to add render tasks in scene
 *     5. allow user to add lights in scene, a point light positioned at (500,500,500) is provided in default
 *        the properties of the lights can be edited 
 *     6. enable/disable display frame-rate
 *     7. save screen capture to file
 * Advanced features:
 *     1. support user defined  custom callback functions
 *     2. support direct operations on the camera, render manager, and light manager
 * Usage:
 *     1. Define a GlutWindow object
 *     2. Set the camera parameters
 *     3. Add render tasks
 *     4. Call createWindow() to create a window
 *     5. Call closewWindow() or click the 'X' on window to close the window
 * Note on defining custom callback functions:
 *     It is quite often that we need to access the GlutWindow object in our custom callback functions, it could be
 *     achieved with one line of code because the GlutWindow object has been binded to glut window in createWindow().
 *     GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
 * Default mouse && keyboard behavior:
 *     Mouse:
 *           1. Left button: rotate camera
 *           2. Middle button: zoom in/out
 *           3. Right button: translate camera
 *     Keyboard:
 *           1. ESC: close window
 *           2. f: enable/disable display frame-rate
 *           3. s: capture screen
 *     The default keyboard behaviors can be retained in your custom keyboard callback if bindDefaultKeys(key,x,y)
 *     is called AT THE BEGINING OF your custom callback with the same arguments as your callback.
 *     We strongly suggest not to override the default key behaviors.
 */

class RenderSceneConfig;

class GlutWindow : public BaseWindow
{
public:
    GlutWindow();                                                                        //initialize a window with default name and size
    GlutWindow(const std::string &window_name);                                          //initialize a window with given name and default size
    GlutWindow(const std::string &window_name, unsigned int width, unsigned int height); //initialize a window with given name and size
    ~GlutWindow();

    void createWindow(); //create window with the parameters set
    void closeWindow();  //close window
    const std::string& name() const;
    int width() const;
    int height() const;

    //getter&&setter of background color and text color
    //Note on getters:
    //1. "window.template xxxColor<ColorType>()" if template paramter of window is unknown
    //2. "window.xxxColor<ColorType>()" is OK as well if template paramter of window is known
    //3. The first use (with template keyword) is always right
    template <typename ColorType> 
    Color<ColorType> backgroundColor() const;
    template <typename ColorType> 
    Color<ColorType> textColor() const;
    template <typename ColorType> 
    void setBackgroundColor(const Color<ColorType> &color);
    template <typename ColorType>
    void setTextColor(const Color<ColorType> &color);

    void enableEventMode();
    void disableEventMode();

    //save screenshot to file
    bool saveScreen(const std::string &file_name) const;  //save to file with given name
    bool saveScreen();                                    //save to file with default name "screen_capture_XXX.png"

    //display frame-rate
    void displayFrameRate() const;  
    void enableDisplayFrameRate();
    void disableDisplayFrameRate();

    //advanced: 
    //set custom callback functions
    void setDisplayFunction(void (*func)(void));  
    void setIdleFunction(void (*func)(void));  
    void setReshapeFunction(void (*func)(int width, int height));
    void setKeyboardFunction(void (*func)(unsigned char key, int x, int y));
    void setSpecialFunction(void (*func)(int key, int x, int y));
    void setMotionFunction(void (*func)(int x, int y));
    void setMouseFunction(void (*func)(int button, int state, int x, int y));
    void setMouseWheelFunction(void(*func)(int wheel, int direction, int x, int y));
    void setInitFunction(void (*func)(void)); //the init function before entering mainloop

    static void bindDefaultKeys(unsigned char key, int x, int y);  //bind the default keyboard behaviors
    
protected:
    //default callback functions
    static void displayFunction(void);                                       //display all render tasks provided by user
    static void idleFunction(void);                                          //do nothing
    static void reshapeFunction(int width, int height);                      //adjust view port to reveal the change
    static void keyboardFunction(unsigned char key, int x, int y);           //press 'ESC' to close window, ect.
    static void specialFunction(int key, int x, int y);                      //do nothing
    static void motionFunction(int x, int y);                                //left button: rotate, middle button: zoom, right button: translate
    static void mouseFunction(int button, int state, int x, int y);          //keep track of mouse state
    static void mouseWheelFunction(int wheel, int direction, int x, int y);  //mouse wheel: zoom
    static void initFunction(void);                                          //init viewport and background color

    void initOpenGLContext();
    void initCallbacks();    //init default callbacks
    void resetMouseState();  //rest mouse
    void initDefaultLight(); //init a default light

protected:
    //pointers to callback methods
    void(*display_function_)(void);
    void(*idle_function_)(void);
    void(*reshape_function_)(int width, int height);
    void(*keyboard_function_)(unsigned char key, int x, int y);
    void(*special_function_)(int key, int x, int y);
    void(*motion_function_)(int x, int y);
    void(*mouse_function_)(int button, int state, int x, int y);
    void(*mouse_wheel_function_)(int wheel, int direction, int x, int y);
    void(*init_function_)(void);

protected:
    //basic information of window
    std::string window_name_;
    int window_id_;

    unsigned int initial_width_;
    unsigned int initial_height_;
    Color<double> background_color_; //use double type in order not to make GlutWindow template
    Color<double> text_color_;       //the color to display text, e.g. fps

    //state of the mouse
    bool left_button_down_, middle_button_down_, right_button_down_;
    int mouse_position_[2];

    //fps display
    bool display_fps_;

    //event mode
    bool event_mode_;
    
    //current screen capture file index
    unsigned int screen_capture_file_index_;
};


template <typename ColorType>
Color<ColorType> GlutWindow::backgroundColor() const
{
    return background_color_.convertColor<ColorType>();
}

template <typename ColorType>
Color<ColorType> GlutWindow::textColor() const
{
    return text_color_.template convertColor<ColorType>();
}

template <typename ColorType>
void GlutWindow::setBackgroundColor(const Color<ColorType> &color)
{
    background_color_ = color.template convertColor<double>();
}

template <typename ColorType>
void GlutWindow::setTextColor(const Color<ColorType> &color)
{
    text_color_ = color.template convertColor<double>();
}

}  //end of namespace Physika

#endif  //PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_
