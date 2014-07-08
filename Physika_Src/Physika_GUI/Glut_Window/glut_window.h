/*
 * @file glut_window.h 
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

#ifndef PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_
#define PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_

#include <string>
#include <GL/gl.h>
#include "Physika_Render/Color/color.h"
#include "Physika_Render/Camera/camera.h"
#include "Physika_Render/Render_Manager/render_manager.h"
#include "Physika_Render/Lights/light_manager.h"
#include "Physika_Render/Lights/light.h"

namespace Physika{

/*
 * Glut-based window
 * Basic features:
 *     1. closing the window will not terminate the program
 *     2. provide default callback functions (see the comments of default functions to view their functionality)
 *     3. provide camera set up
 *     4. allow user to add render tasks in scene
 *     5. allow user to add lights in scene, a point light positioned at origion is provided in default 
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

class GlutWindow
{
public:
    GlutWindow();  //initialize a window with default name and size
    GlutWindow(const std::string &window_name); //initialize a window with given name and default size
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

    //camera operations
    const Vector<double,3>& cameraPosition() const;
    void setCameraPosition(const Vector<double,3> &position);
    const Vector<double,3>& cameraUpDirection() const;
    void setCameraUpDirection(const Vector<double,3> &up);
    const Vector<double,3>& cameraFocusPosition() const;
    void setCameraFocusPosition(const Vector<double,3> &focus);
    double cameraFOV() const;
    void setCameraFOV(double fov);
    double cameraAspect() const;
    void setCameraAspect(double aspect);
    double cameraNearClip() const;
    void setCameraNearClip(double near_clip);
    double cameraFarClip() const;
    void setCameraFarClip(double far_clip);
    void orbitCameraUp(double rad);
    void orbitCameraDown(double rad);
    void orbitCameraLeft(double rad);
    void orbitCameraRight(double rad);
    void zoomCameraIn(double dist);
    void zoomCameraOut(double dist);
    void yawCamera(double rad);
    void pitchCamera(double rad);
    void rollCamera(double rad);
    void translateCameraUp(double dist);
    void translateCameraDown(double dist);
    void translateCameraLeft(double dist);
    void translateCameraRight(double dist);

    //manages lights
    unsigned int numLights() const;
    void pushBackLight(Light*);
    void pushFrontLight(Light*);
    void insertLightAtIndex(unsigned int index,Light *light);
    void popBackLight();
    void popFrontLight();
    void removeLightAtIndex(unsigned int index);
    void removeAllLights();
    const Light* lightAtIndex(unsigned int index) const;
    Light* lightAtIndex(unsigned int index);
    int lightIndex(Light *light) const;   //return index of light in list, if light not in list ,return -1

    //manages render tasks
    unsigned int numRenderTasks() const;  //length of the render queue
    void pushBackRenderTask(RenderBase*);  //insert new task at back of render queue
    void pushFrontRenderTask(RenderBase*); //insert new task at front of render queue
    void insertRenderTaskAtIndex(unsigned int index,RenderBase *task);  //insert new task before the index-th task
    void popBackRenderTask(); //remove task at back of render queue
    void popFrontRenderTask();  //remove task at front of render queue
    void removeRenderTaskAtIndex(unsigned int index);  //remove the index-th task in queue
    void removeAllRenderTasks();  //remove all render tasks
    const RenderBase* getRenderTaskAtIndex(unsigned int index) const; //return pointer to the render task at given index
    RenderBase* getRenderTaskAtIndex(unsigned int index);
    int getRenderTaskIndex(RenderBase *task) const; //return index of task in queue, if task not in queue, return -1

    //save screenshot to file
    bool saveScreen(const std::string &file_name) const;  //save to file with given name
    bool saveScreen() const; //save to file with default name "screen_capture_XXX.png"
    //display frame-rate
    void displayFrameRate() const;  //display framerate if enabled
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
    void setInitFunction(void (*func)(void)); //the init function before entering mainloop
    static void bindDefaultKeys(unsigned char key, int x, int y);  //bind the default keyboard behaviors
    //direct operation on camera, render manager, and light manager
    const Camera<double>& camera() const{ return camera_;}
    Camera<double>& camera() { return camera_;}
    const RenderManager& renderManager() const{ return render_manager_;}
    RenderManager& renderManager() { return render_manager_;}
    const LightManager& lightManager() const{ return light_manager_;}
    LightManager& lightManager() { return light_manager_;}
    //apply camera and lights: call this method in your custom display method before the rendering code 
    //such that the camera and lights are work as you set
    void applyCameraAndLights();
protected:
    //default callback functions
    static void displayFunction(void);  //display all render tasks provided by user
    static void idleFunction(void);  //do nothing
    static void reshapeFunction(int width, int height);  //adjust view port to reveal the change
    static void keyboardFunction(unsigned char key, int x, int y);  //press 'ESC' to close window, ect.
    static void specialFunction(int key, int x, int y);  //do nothing
    static void motionFunction(int x, int y);  //left button: rotate, middle button: zoom, right button: translate
    static void mouseFunction(int button, int state, int x, int y);  //keep track of mouse state
    static void initFunction(void);  // init viewport and background color

    void initCallbacks();  //init default callbacks
    void resetMouseState();  //rest mouse
    void initDefaultLight(); //init a default light
protected:
    //basic information of window
    std::string window_name_;
    int window_id_;
    unsigned int initial_width_;
    unsigned int initial_height_;
    Color<double> background_color_; //use double type in order not to make GlutWindow template

    //camera (use double type in order not to make GlutWindow template)
    Camera<double> camera_;
    //render managner, manages the scene for render
    RenderManager render_manager_;
    //light manager, manages the lights in scene
    LightManager light_manager_;
    Light default_light_;  //the default light
    //state of the mouse
    bool left_button_down_, middle_button_down_, right_button_down_;
    int mouse_position_[2];
    //fps display
    bool display_fps_;
    Color<double> text_color_; //the color to display text, e.g. fps
    //pointers to callback methods
    void (*display_function_)(void);
    void (*idle_function_)(void);
    void (*reshape_function_)(int width, int height);
    void (*keyboard_function_)(unsigned char key, int x, int y);
    void (*special_function_)(int key, int x, int y);
    void (*motion_function_)(int x, int y);
    void (*mouse_function_)(int button, int state, int x, int y);
    void (*init_function_)(void);
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
