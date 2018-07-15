/******************************************************************************
Copyright (c) 2007 Bart Adams (bart.adams@gmail.com)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software. The authors shall be
acknowledged in scientific publications resulting from using the Software
by referencing the ACM SIGGRAPH 2007 paper "Adaptively Sampled Particle
Fluids".

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
******************************************************************************/

#ifndef __CANVAS_H__
#define __CANVAS_H__

#include "IWindow.h"
#include "Timer.h"

/**
  * GlutWindow class used to draw on.
  * Inherits from GlutWindow.
  */
class GlutWindow : public IWindow {
public:
    // constructor
    GlutWindow(char * file, int width=512, int height=512);
    // destructor
    virtual ~GlutWindow();

    void setBackgroundColor(const Vectorold3f color);

    // inherited from GlutWindow
    virtual void CallBackMouseFunc(int button, int state, int x, int y);
    virtual void CallBackDisplayFunc();
    virtual void CallBackMotionFunc(int x, int y);
    virtual void CallBackKeyboardFunc(unsigned char key, int x, int y);
    virtual void CallBackIdleFunc(void);
    virtual void CallBackReshapeFunc(int w, int h);
	virtual void CallBackSpecialFunc(int key, int x, int y);
	virtual void CallBackVisibilityFunc(int visible);
	virtual void CallBackPassiveMotionFunc(int x, int y);

    void drawString(std::string s, Vectorold3f color, int x, int y);

    void saveImage(int iter=0) const;

	void InitModelView();

private:
    void initViewerShape(int width, int height);
    void initViewerOrientation();
    void displayFPS(float cost);
    void displayHelp();

    int _width;
    int _height;
    Vectorold3f _backgroundcolor;

    CTimer _timer;

    bool _animate;
    int _totalmsec;
    bool _fps;
    bool _help;

    int _button;

    int _iter;
    int _dumpiter;
    bool _dumpImages;
    int _time, _timebase, _frame;
    float fps;
};

#endif
