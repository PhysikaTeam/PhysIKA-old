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
#include "Physika_GUI/Camera/camera.h"
#include "Physika_Render/Render_Manager/render_manager.h"

namespace Physika{

/*
 * Glut-based window
 * Key features:
 *       1. provide default response functions, and support custom response functions
 *          see the comments of default response functions to view their functionality
 *       2. closing the window will not close the program
 *  
 * Usage:
 *       1. Define a GlutWindow object
 *       2. Set the custom callback functions (optional)
 *       3. Call createWindow() 
 *       4. Call closeWindow() or click the 'X' on window to close the window
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
    void translateCamera(const Vector<double,3> &vec);

    //manages render tasks
    unsigned int numRenderTasks() const;  //length of the render queue
    void pushBackRenderTask(RenderBase*);  //insert new task at back of render queue
    void pushFrontRenderTask(RenderBase*); //insert new task at front of render queue
    void insertRenderTaskAtIndex(unsigned int index,RenderBase *task);  //insert new task before the index-th task
    void popBackRenderTask(); //remove task at back of render queue
    void popFrontRenderTask();  //remove task at front of render queue
    void removeRenderTaskAtIndex(unsigned int index);  //remove the index-th task in queue
    void removeAllRenderTasks();  //remove all render tasks
    const RenderBase* renderTaskAtIndex(unsigned int index) const; //return pointer to the render task at given index
    RenderBase* renderTaskAtIndex(unsigned int index);
    int renderTaskIndex(RenderBase *task) const; //return index of task in queue, if task not in queue, return -1

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
    //direct operation on camera and render manager
    const Camera<double>& camera() const{ return camera_;}
    Camera<double>& camera() { return camera_;}
    const RenderManager& renderManager() const{ return render_manager_;}
    RenderManager& renderManager() { return render_manager_;}
protected:
    //default callback functions
    static void displayFunction(void);  //display background color
    static void idleFunction(void);  //do nothing
    static void reshapeFunction(int width, int height);  //adjust view port to reveal the change
    static void keyboardFunction(unsigned char key, int x, int y);  //press 'ESC' to close window
    static void specialFunction(int key, int x, int y);  //do nothing
    static void motionFunction(int x, int y);  //do nothing
    static void mouseFunction(int button, int state, int x, int y);  //do nothing
    static void initFunction(void);  // init viewport and background color
    //init default callbacks
    void initCallbacks();
protected:
    //basic information of window
    std::string window_name_;
    int window_id_;
    unsigned int initial_width_;
    unsigned int initial_height_;
    //camera (use double type in order not to make GlutWindow template)
    Camera<double> camera_;
    //render managner, manages the scene for render
    RenderManager render_manager_;
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

}  //end of namespace Physika

#endif  //PHYSIKA_GUI_GLUT_WINDOW_GLUT_WINDOW_H_
