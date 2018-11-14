/*
 * @file glut_window.cpp 
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

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <GL/glew.h>
#include <GL/freeglut.h>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Core/Utilities/physika_exception.h"
#include "Physika_Core/Image/image.h"
#include "Physika_IO/Image_IO/image_io.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

namespace gui{

GlutWindow::GlutWindow()
    :window_name_(std::string("Physika Glut Window")),window_id_(-1),initial_width_(640),initial_height_(480),
     display_fps_(true),screen_capture_file_index_(0),event_mode_(false)
{
    background_color_ = Color<double>(0.5, 0.5, 0.5, 1.0);
    text_color_ = Color<double>::White();

    resetMouseState();
    initCallbacks();
    initOpenGLContext();

//     RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
//     render_scene_config.setCameraAspect(static_cast<double>(initial_width_ / initial_height_));
// 
//     //reset screen based render manager
//     ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();
//     render_manager.resetMsaaFBO(initial_width_, initial_height_);
}

GlutWindow::GlutWindow(const std::string &window_name)
    :window_name_(window_name),window_id_(-1),initial_width_(640),initial_height_(480),
     display_fps_(true),screen_capture_file_index_(0), event_mode_(false)
{
    background_color_ = Color<double>::Black();
    text_color_ = Color<double>::White();

    resetMouseState();
    initCallbacks();
    initOpenGLContext();

//     RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
//     render_scene_config.setCameraAspect(static_cast<double>(initial_width_ / initial_height_));
// 
//     //reset screen based render manager
//     ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();
//     render_manager.resetMsaaFBO(initial_width_, initial_height_);
}

GlutWindow::GlutWindow(const std::string &window_name, unsigned int width, unsigned int height)
    :window_name_(window_name),window_id_(-1),initial_width_(width),initial_height_(height),
     display_fps_(true),screen_capture_file_index_(0), event_mode_(false)
{
    background_color_ = Color<double>::Black();
    text_color_ = Color<double>::White();

    resetMouseState();
    initCallbacks();
    initOpenGLContext();

//     RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
//     render_scene_config.setCameraAspect(static_cast<double>(initial_width_ / initial_height_));
// 
//     //reset screen based render manager
//     ScreenBasedRenderManager & render_manager = render_scene_config.screenBasedRenderManager();
//     render_manager.resetMsaaFBO(initial_width_, initial_height_);
}

GlutWindow::~GlutWindow()
{
}

void GlutWindow::createWindow()
{
    glutShowWindow();
    glutSetWindowData(this);  //bind 'this' pointer with the window
    glutDisplayFunc(display_function_);
    glutIdleFunc(idle_function_);
    glutReshapeFunc(reshape_function_);
    glutKeyboardFunc(keyboard_function_);
    glutSpecialFunc(special_function_);
    glutMotionFunc(motion_function_);
    glutMouseFunc(mouse_function_);
    glutMouseWheelFunc(mouse_wheel_function_);

    resetMouseState(); //reset the state of mouse every time the window is created

    (*init_function_)(); //call the init function before entering main loop

	m_camera.RegisterPoint(0.5f, 0.5f);
	m_camera.TranslateToPoint(0, 0);

	m_camera.Zoom(3.0);
	m_camera.SetGL(0.01f, 10.0f, width(), height());
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

void GlutWindow::enableEventMode()
{
    this->event_mode_ = true;
}

void GlutWindow::disableEventMode()
{
    this->event_mode_ = false;
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

void GlutWindow::displayFrameRate()
{
    if(!glutGet(GLUT_INIT_STATE))  //window is not created
        throw PhysikaException("Cannot display frame rate before a window is created.");
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
       
		drawString(str, Color<float>(1.0f, 1.0f, 1.0f), 0, height()-25);
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

void GlutWindow::setMouseWheelFunction(void(*func)(int wheel, int direction, int x, int y))
{
    if (func == NULL)
    {
        std::cerr << "NULL callback function provided, use default instead.\n";
        mouse_wheel_function_ = GlutWindow::mouseWheelFunction;
    }
    else
        mouse_wheel_function_ = func;
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

void GlutWindow::mainLoop()
{
	if (event_mode_ == false)
		glutMainLoop();
}

////////////////////////////////////////////////// default callback functions ////////////////////////////////////////////////////////////////////

void GlutWindow::displayFunction(void)
{
    GlutWindow * cur_window = (GlutWindow*)glutGetWindowData();
    Color<double> background_color = cur_window->background_color_;

    glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1, 0, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(1, 0, 0);
	glColor3f(0, 1, 0);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 1, 0);
	glColor3f(0, 0, 1);
	glVertex3f(0, 0, 0);
	glVertex3f(0, 0, 1);
	glEnd();

	cur_window->getScene()->draw();

    cur_window->displayFrameRate();
//    glutPostRedisplay();
    glutSwapBuffers();
}

void GlutWindow::idleFunction(void)
{
	GlutWindow * cur_window = (GlutWindow*)glutGetWindowData();
	cur_window->getScene()->takeOneFrame();

    glutPostRedisplay();
}

void GlutWindow::reshapeFunction(int width, int height)
{
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
	window->activeCamera().SetGL(0.01f, 10.0f, width, height);
	glViewport(0, 0, width, height);
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
	Camera& activeCamera = window->activeCamera();

	if (window->getButtonType() == GLUT_LEFT_BUTTON) {
		activeCamera.RotateToPoint(float(x) / float(window->width()) - 0.5f, float(window->height() - y) / float(window->height()) - 0.5f);
	}
	else if (window->getButtonType() == GLUT_RIGHT_BUTTON) {
		activeCamera.TranslateToPoint(float(x) / float(window->width()) - 0.5f, float(window->height() - y) / float(window->height()) - 0.5f);
	}
	else if (window->getButtonType() == GLUT_MIDDLE_BUTTON) {
		activeCamera.TranslateLightToPoint(float(x) / float(window->width()) - 0.5f, float(window->height() - y) / float(window->height()) - 0.5f);
	}
	activeCamera.SetGL(0.01f, 10.0f, window->width(), window->height());
	glutPostRedisplay();
}

void GlutWindow::mouseFunction(int button, int state, int x, int y)
{
     GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
	 Camera& activeCamera = window->activeCamera();
	 window->setButtonType(button);
	 window->setButtonState(state);

	if (state == GLUT_DOWN) {
		activeCamera.RegisterPoint(float(x) / float(window->width()) - 0.5f, float(window->height() - y) / float(window->height()) - 0.5f);
	}
}

void GlutWindow::mouseWheelFunction(int wheel, int direction, int x, int y)
{
	GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
	Camera& activeCamera = window->activeCamera();

	switch (direction)
	{
	case 1:
		activeCamera.Zoom(-0.3);
		activeCamera.SetGL(0.01f, 10.0f, window->width(), window->height());
		break;
	case -1:
		activeCamera.Zoom(0.3);
		activeCamera.SetGL(0.01f, 10.0f, window->width(), window->height());
	default:
		break;
	}
}

void GlutWindow::initFunction(void)
{
    int width = glutGet(GLUT_WINDOW_WIDTH);
    int height = glutGet(GLUT_WINDOW_HEIGHT);
    GlutWindow *window = static_cast<GlutWindow*>(glutGetWindowData());
    PHYSIKA_ASSERT(window);

    glViewport(0, 0, width, height);        									// set the viewport
    window->initDefaultLight();

    glShadeModel( GL_SMOOTH );
    glClearDepth( 1.0 );														// specify the clear value for the depth buffer
    glEnable( GL_DEPTH_TEST );
    
    glHint( GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST );						// specify implementation-specific hints
    Color<double> background_color = window->background_color_;
    
	glClearColor(background_color.redChannel(), background_color.greenChannel(), background_color.blueChannel(), background_color.alphaChannel());
}

void GlutWindow::initOpenGLContext()
{
    int argc = 1;
    const int max_length = 1024; //assume length of the window name does not exceed 1024 characters
    char *argv[1];
    char name_str[max_length];
    strcpy(name_str, window_name_.c_str());
    argv[0] = name_str;

    glutInit(&argc, argv);
    glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);  //this option allows leaving the glut loop without exit program
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_ALPHA);
    glutInitWindowSize(initial_width_, initial_height_);
    window_id_ = glutCreateWindow(window_name_.c_str());
    glutHideWindow();

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK)
    {
        std::cerr << "error: can't init glew!\n";
        std::exit(EXIT_FAILURE);
    }

    std::cout << "openGL Version: " << glGetString(GL_VERSION) << std::endl;

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
    mouse_wheel_function_ = GlutWindow::mouseWheelFunction;
    init_function_ = GlutWindow::initFunction;
}

void GlutWindow::resetMouseState()
{
}

void GlutWindow::initDefaultLight()
{
    std::cout << "info: add default flash light!" << std::endl;

//     std::shared_ptr<FlashLight> flash_light = std::make_shared<FlashLight>();
//     flash_light->setAmbient(Color4f::Gray());
// 
//     RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
//     render_scene_config.pushBackLight(std::move(flash_light));
}

void GlutWindow::drawString(std::string s, Color<float> &color, int x, int y)
{
	glPushAttrib(GL_ALL_ATTRIB_BITS);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadIdentity();
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	gluOrtho2D(0, width(), 0, height());
	glColor3f(color.redChannel(), color.greenChannel(), color.blueChannel());
	glRasterPos2f(x, y);

	for (int i = 0; i < (int)s.length(); i++) {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, s[i]);
	}

	glPopMatrix();
	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();
	glEnable(GL_LIGHTING);

	glPopAttrib();
}

} //end of namespace Physika
