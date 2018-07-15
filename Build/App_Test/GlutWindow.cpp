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

#include "GlutWindow.h"
#include <iostream>
#include <string>
#include <sstream>
#include "Image.h"
using namespace std;

GlutWindow::GlutWindow(char * file, int width, int height) {
	// Create dummy variables

	char * dummy_argv[1];
	dummy_argv[0] = "run";
	int dummy_argc = 1;

	// Initialize GLUT

	glutInit(&dummy_argc, dummy_argv);

    setBackgroundColor(Vectorold3f(0.3,0.3,0.3));
    glutInitDisplayMode (GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glutInitWindowSize(width,height);

	// Open new window, record its windowID ,
	windowID = glutCreateWindow(file);

	if (glewInit() != GLEW_OK)
	{
		std::cout << "Initialization failed!" << std::endl;
	}

	if (!glewIsSupported("GL_VERSION_2_0 "
		"GL_VERSION_1_5 "
		"GL_ARB_multitexture "
		"GL_ARB_vertex_buffer_object")) 
	{
		std::cout << "VOB is not supported!" << std::endl;
	}
//     GlutMaster::instance()->CallGlutCreateWindow("", this);
//     GlutMaster::instance()->SetIdleToCurrentWindow();

    _iter = 0;
    _dumpiter = 1000000;
    _dumpImages = false;
	
    initViewerShape(width, height);
    initViewerOrientation();

    _camera.RegisterPoint(0.5f, 0.5f);
    _camera.TranslateToPoint(0,0);

    _camera.Zoom(3.0);
    _camera.SetGL(0.01f, 10.0f, _width, _height);

    _animate = false; // standard no rotation
    _totalmsec = 1; // 10 seconds for one rotation
    _fps = true;
    _help = true;
    fps = 0;

    _time=_timebase=_frame=0;

	InitModelView();
}

GlutWindow::~GlutWindow() {
}

void GlutWindow::initViewerShape(int width, int height) {
    _width = width;
    _height = height;
    _camera.SetGL(0.01f, 10.0f, width, height);
    glViewport(0,0,width, height);
}

void GlutWindow::initViewerOrientation() {
}

void GlutWindow::setBackgroundColor(const Vectorold3f color) {
    _backgroundcolor = color;
}

void GlutWindow::CallBackMouseFunc(int button, int state, int x, int y) {
    if (state == GLUT_DOWN) {
        _button = button;
        _camera.RegisterPoint(float(x)/float(_width)-0.5f,float(_height-y)/float(_height)-0.5f);
    }
    else if (button==3) { // don't know what the GLUT_... for this is
        _camera.Zoom(-0.3);
        _camera.SetGL(0.01f, 10.0f, _width, _height);
    }
    else if (button==4) { // idem
        _camera.Zoom(0.3);
        _camera.SetGL(0.01f, 10.0f, _width, _height);
    }
    glutPostRedisplay();
}

void GlutWindow::displayFPS(float cost) {
	if (_fps)
	{
		//     if (_fps) {
		// 	   _frame++;
		// 	   _time=glutGet(GLUT_ELAPSED_TIME);
		// 	   if (_time - _timebase > 10) {
		// 		   fps= _frame*1000.0/(_time-_timebase);
		// 		   _timebase = _time;		
		// 		   _frame = 0;
		// 	   }

		if (cost > 1)
		{
			fps = 1000.0/cost;
		}
		else
		{
			fps = 1000.0f;
		}
		
		stringstream stream;
		if (fps>1) {
			stream.precision(2);
			stream << (fps);
			string s = stream.str();
			string s2 = string("fps: ") +s;
			drawString(s2, Vectorold3f(1.0,0.0,0.0),5,_height-19);
		}
		else {
			stream.precision(2);
			stream << (1.0f/fps);
			string s = stream.str();
			string s2 = string("spf: ") +s;
			drawString(s2, Vectorold3f(1.0,0.0,0.0),5,_height-19);
		}
	}     
}

void GlutWindow::saveImage(int iter) const {
    stringstream stream;
    stream << iter;
    string s = string("images/image_") + stream.str() + string(".ppm");
    
	GLubyte* the_image = new GLubyte[_height*_width*3];
    //orgin//GLubyte the_image[_height][_width][3];
    // 0 = red, 1 = green, 2 = blue ranging from 0 to 255.
    glFlush();
    glReadPixels(0,0,_width, _height, GL_RGB, GL_UNSIGNED_BYTE, (void*)the_image);
    Image image(_width, _height);
    for (int i=0; i<_width; i++) {
        for (int j=0; j<_height; j++) {
            image.setPixel(i,_height-1-j,0,float(the_image[(j*_width+i)*3+0])/255.0f);
            image.setPixel(i,_height-1-j,1,float(the_image[(j*_width+i)*3+1])/255.0f);
            image.setPixel(i,_height-1-j,2,float(the_image[(j*_width+i)*3+2])/255.0f);
        }
    }
    image.save(s.c_str());
    string convert = string("convert ") + s + string(" ") + s + string(".png && rm ") + s;
    system(convert.c_str());
	delete []the_image;
}

void GlutWindow::displayHelp() {
}

void GlutWindow::CallBackDisplayFunc() {


    glClearColor(_backgroundcolor.x, _backgroundcolor.y, _backgroundcolor.z,1);
    glClear (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode (GL_MODELVIEW);
    //_renderer->render(_camera);

	glDisable(GL_LIGHTING);
	glLineWidth(4);
	glBegin(GL_LINES);
	glColor3f(1,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1,0,0);
	glColor3f(0,1,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1,0);
	glColor3f(0,0,1);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1);
	glEnd();

	DrawScene();

    if (_help) {
        displayHelp();
    }

	displayFPS(_totalmsec);

	glutSwapBuffers();

	glutPostRedisplay();
}

void GlutWindow::CallBackMotionFunc(int x, int y) {
    if (_button == GLUT_LEFT_BUTTON) {
        _camera.RotateToPoint(float(x)/float(_width)-0.5f, float(_height-y)/float(_height)-0.5f);
    }
    else if (_button == GLUT_RIGHT_BUTTON) {
        _camera.TranslateToPoint(float(x)/float(_width)-0.5f, float(_height-y)/float(_height)-0.5f);
    }
    else if (_button == GLUT_MIDDLE_BUTTON) {
        _camera.TranslateLightToPoint(float(x)/float(_width)-0.5f, float(_height-y)/float(_height)-0.5f);
    }
    _camera.SetGL(0.01f, 10.0f, _width, _height);
    glutPostRedisplay();
}

void GlutWindow::CallBackKeyboardFunc(unsigned char key, int x, int y) {
    cout << "Key Pressed: " << key << endl;
    switch (key) {
    case '4' : {
            break;
        }
	case 'm':
		{
			ShiftRenderingMode();
			break;
		}
		
    case 'a' : {
            _animate=!_animate;
            break;
        }
//    case 'f' : glutFullScreen(); break;
//    case 'f' : _fps=!_fps; break;
//    case 'r' : initViewerOrientation(); break;
//    case 's' : saveImage(); break;
//    case 'l' : _scene.toggleLighting(); break;
    case 'q' : exit(0); break;
    case 'Q' : exit(0); break;
    case 'h' : _help=!_help; break;
    case 'b': {
            // FIXME: check input!
            cout << "Give background rgb int triplet (range 0..255):" << endl;
            cout << "Red: "; int red; cin >> red;
            cout << "Green: "; int green; cin >> green;
            cout << "Blue: "; int blue; cin >> blue;
            setBackgroundColor(Vectorold3f(float(red)/255.0f, float(green)/255.0f, float(blue)/255.0f));
            break;
        }
	case 'j':{ // don't know what the GLUT_... for this is
			_camera.Zoom(-0.3);
			_camera.SetGL(0.01f, 10.0f, _width, _height);
		}
			 break;
	case 'k':{ // idem
			_camera.Zoom(0.3);
			_camera.SetGL(0.01f, 10.0f, _width, _height);
		}
			 break;

    default : 
		break;
    }
    
	_sim->invoke('K', key, x, y);

	glutPostRedisplay();
}

void GlutWindow::CallBackIdleFunc(void) {
	if (_animate) {
		_sim->takeOneFrame();
		_iter++;
	}

	_totalmsec = _sim->getTimeCostPerFrame();
}

void GlutWindow::CallBackReshapeFunc(int w, int h) {
    cout << "widthxheight: " << w << "x"<<h << endl;
    initViewerShape(w,h);
    _camera.SetGL(0.01f, 10.0f, w, h);
    glutPostRedisplay();
}

void GlutWindow::CallBackSpecialFunc( int key, int x, int y )
{

}

void GlutWindow::CallBackVisibilityFunc( int visible )
{

}

void GlutWindow::CallBackPassiveMotionFunc( int x, int y )
{

}


// --------------------------------------------------------------------------------//
// STRING DRAWING UTILITY FUNCTION
void GlutWindow::drawString(std::string s, Vectorold3f color, int x, int y) {
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();
    glLoadIdentity();
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    gluOrtho2D(0,_width,0,_height);
    glColor3f(color.x, color.y, color.z);
    glRasterPos2f(x,y);

    for (int i = 0; i < (int)s.length(); i++) {
        glutBitmapCharacter (GLUT_BITMAP_HELVETICA_18, s[i]);
    }

    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glEnable(GL_LIGHTING);
}

void GlutWindow::InitModelView()
{
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	GLfloat shin[] = {100.0f};
	GLfloat diff[] = {0.8f,0.8f,0.8f,1.0f};
	GLfloat amb[] = {1,1,1,1};
	glLightfv(GL_LIGHT0, GL_AMBIENT, amb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, diff);
	glEnable(GL_LIGHT0);

	
	GLfloat mat_ambient[] = {0.7f, 0.7f, 0.7f, 1.0f};
	GLfloat mat_diffuse[] = {0.7f, 0.7f, 0.7f, 1.0f};
	glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient);
	glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse);
	
}








