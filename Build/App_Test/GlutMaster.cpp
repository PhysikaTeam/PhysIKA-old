////////////////////////////////////////////////////////////////
//                                                            //
// glutMaster.c++                                             //
// beta version 0.3 - 9/9/97)                                 //
//                                                            //
// George Stetten and Korin Crawford                          //
// copyright given to the public domain                       //
//                                                            //
// Please email comments to stetten@acpub.duke.edu,           //
//                                                            //
////////////////////////////////////////////////////////////////

#include "glutMaster.h"
#include "glutWindow.h"

GlutWindow * viewPorts[MAX_NUMBER_OF_WINDOWS];

int GlutMaster::currentIdleWindow   = 0;
int GlutMaster::idleFunctionEnabled = 0;
GlutMaster * GlutMaster::_instance = 0;

GlutMaster * GlutMaster::instance() {
    if (_instance==0) {
        _instance = new GlutMaster();
    }
    return _instance;
}


GlutMaster::GlutMaster() {
}

GlutMaster::~GlutMaster() {
}

void GlutMaster::CallBackDisplayFunc(void) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackDisplayFunc();
}

void GlutMaster::CallBackIdleFunc(void) {

    if(idleFunctionEnabled && currentIdleWindow) {
        glutSetWindow(currentIdleWindow);
        viewPorts[currentIdleWindow]->CallBackIdleFunc();
    }
}

void GlutMaster::CallBackKeyboardFunc(unsigned char key, int x, int y) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackKeyboardFunc(key, x, y);
}

void GlutMaster::CallBackMotionFunc(int x, int y) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackMotionFunc(x, y);
}

void GlutMaster::CallBackMouseFunc(int button, int state, int x, int y) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackMouseFunc(button, state, x, y);
}

void GlutMaster::CallBackPassiveMotionFunc(int x, int y) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackPassiveMotionFunc(x, y);
}

void GlutMaster::CallBackReshapeFunc(int w, int h) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackReshapeFunc(w, h);
}

void GlutMaster::CallBackSpecialFunc(int key, int x, int y) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackSpecialFunc(key, x, y);
}

void GlutMaster::CallBackVisibilityFunc(int visible) {

    int windowID = glutGetWindow();
    viewPorts[windowID]->CallBackVisibilityFunc(visible);
}

void GlutMaster::SetActiveWindow(char * setTitle, GlutWindow * glutWindow) {
    // Store the address of new window in global array
    // so GlutMaster can send events to propoer callback functions.

    viewPorts[glutWindow->GetWindowID()] = glutWindow;

    // Hand address of universal static callback functions to Glut.
    // This must be for each new window, even though the address are constant.

    glutDisplayFunc(CallBackDisplayFunc);
    glutIdleFunc(CallBackIdleFunc);
    glutKeyboardFunc(CallBackKeyboardFunc);
    glutMouseFunc(CallBackMouseFunc);
    glutMotionFunc(CallBackMotionFunc);
    glutPassiveMotionFunc(CallBackPassiveMotionFunc);
    glutReshapeFunc(CallBackReshapeFunc);
    glutVisibilityFunc(CallBackVisibilityFunc);
}

void GlutMaster::CallGlutMainLoop(void) {

    glutMainLoop();
}

void GlutMaster::DisableIdleFunction(void) {

    idleFunctionEnabled = 0;
}

void GlutMaster::EnableIdleFunction(void) {

    idleFunctionEnabled = 1;
}

int GlutMaster::IdleFunctionEnabled(void) {

    // Is idle function enabled?

    return(idleFunctionEnabled);
}

int GlutMaster::IdleSetToCurrentWindow(void) {

    // Is current idle window same as current window?

    return( currentIdleWindow == glutGetWindow() );
}

void GlutMaster::SetIdleToCurrentWindow(void) {

    currentIdleWindow = glutGetWindow();
}














