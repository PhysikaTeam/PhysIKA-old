#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif

#include <GL/glh_glut.h>
#include <stdio.h>

bool b[256];
int win_w = 512, win_h = 512;

using namespace glh;
glut_simple_mouse_interactor object;
void CaptureScreen(int, int);

float DISP_SCALE = 0.001f;
char *dataPath;
int stFrame = 0;
float p[3], q[3];

float gDepOff = 0.025;

// for sprintf
#pragma warning(disable: 4996)

extern void initModel(const char *);
extern void initModel(int, char **, char *, int);
void findMatchNodes(const char *ifile, const char *ofile);
void initModel(const char *cfile, const char *ofile);

extern void quitModel();
extern void drawModel(bool, bool, bool, bool, int);
extern void updateModel();
extern bool dynamicModel(char *, bool, bool);
extern void dumpModel();
extern void loadModel();
extern void checkModel();
extern void checkCCD();
extern void checkSelfCPU_Naive();
extern void checkOverlapNodes();
extern void findMatchNodes();
extern void findMatchNodes2();
extern void findMatchByInputID();

extern void checkSelfCPU_Rebuild();
extern void checkSelfCPU_Refit(const char *);
extern void checkSelfGPU();
extern bool checkSelfIJ(int, int);

extern void checkSelfGPU_SH();

static int level = 1;

float lightpos[4] = {13, 10.2, 3.2, 0};

// check for OpenGL errors
void checkGLError()
{
    GLenum error;
    while ((error = glGetError()) != GL_NO_ERROR) {
		char msg[512];
		sprintf(msg, "error - %s\n", (char *) gluErrorString(error));
		printf(msg);
    }
}

void initSetting()
{
	b['9'] = false;
}

void initOpengl()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);

	// initialize OpenGL lighting
	GLfloat lightPos[] =   {10.0, 10.0, 10.0, 0.0};
	GLfloat lightAmb[4] =  {0.0, 0.0, 0.0, 1.0};
	GLfloat lightDiff[4] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lightSpec[4] = {1.0, 1.0, 1.0, 1.0};

	glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
	glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmb);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiff);
	glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpec);

	//glLightModeli(GL_LIGHT_MODEL_COLOR_CONTROL_EXT, GL_SEPARATE_SPECULAR_COLOR_EXT);
	GLfloat black[] =  {0.0, 0.0, 0.0, 1.0};
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, black);
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);

	glEnable(GL_LIGHT0);
	glEnable(GL_LIGHTING);
	glEnable(GL_DEPTH_TEST);
}

void updateFPS()
{
}

void begin_window_coords()
{
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(0.0, win_w, 0.0, win_h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void end_window_coords()
{
    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
}

void drawGround()
{
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

	glBegin(GL_QUADS);
	glColor3f(1.f, 0.f, 0.f);
	glVertex3f(20, 0, 20);
	glVertex3f(-20, 0, 20);
	glVertex3f(-20, 0, -20);
	glVertex3f(20, 0, -20);
	glEnd();

	glDisable(GL_COLOR_MATERIAL);
}

extern void drawEdges(bool, bool);
extern void drawVFs(int);
extern void drawDebugVF(int);

void drawOther()
{
	glDisable(GL_LIGHTING);

	glEnable(GL_LIGHTING);
}


void draw()
{
#ifdef DRAW_VF
	drawDebugVF(level);
	//drawVFs(0);
#else
#ifdef DRAW_EDGE
	drawEdges(b['t'], b['s']);
#else
	glPushMatrix();
	glRotatef(-90, 1, 0, 0);

	drawModel(!b['t'], !b['p'], !b['s'], b['e'], level);

	/*	glPushMatrix();
	glTranslatef(0, 0, -0.5f);
	glutSolidSphere(0.3, 20, 20);
	glPopMatrix();
*/
	glPopMatrix();
#endif
#endif

	if (b['g'])
		drawGround();
}

static bool ret = false;

void display()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glDisable(GL_DEPTH_TEST);
	glDisable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	glShadeModel(GL_SMOOTH);

	if (!b['b']) {
    // gradient background
    begin_window_coords();
    glBegin(GL_QUADS);
        glColor3f(0.2, 0.4, 0.8);
        glVertex2f(0.0, 0.0);
        glVertex2f(win_w, 0.0);
        glColor3f(0.05, 0.1, 0.2);
        glVertex2f(win_w, win_h);
        glVertex2f(0, win_h);
    glEnd();
    end_window_coords();
	}

    glMatrixMode(GL_MODELVIEW);

//#define LOAD_VIEW
#ifdef LOAD_VIEW
	static bool load = true;
	static GLdouble modelMatrix[16];
	if (load) {
		FILE *fp = fopen("c:\\temp\\view-ro.dat", "rb");
		fread(modelMatrix, sizeof(GLdouble), 16, fp);
		fclose(fp);
		load = false;
	}
	glLoadMatrixd(modelMatrix);
#else
	glLoadIdentity();
    object.apply_transform();

	if (b['v']	) {
		GLdouble modelMatrix[16];
		glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
		FILE *fp=fopen("c:\\temp\\view.dat", "wb");
		fwrite(modelMatrix, sizeof(GLdouble), 16, fp);
		fclose(fp);
		b['v'] = false;
	}
#endif

	// draw scene
	if (b['w'])
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // draw scene
/*	if (b['l'])
		glDisable(GL_LIGHTING);
	else
*/
		glEnable(GL_LIGHTING);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);

	draw();

    glutSwapBuffers();
	updateFPS();
	//checkGLError();

	if(b['x'] && ret)   {
		CaptureScreen(512, 512);
	}
}

void key5();
void key4();
void key6();

void idle()
{
    if (b[' '])
        object.trackball.increment_rotation();

	if (b['d']) {
		ret = dynamicModel(dataPath, b['o'], false);
		//key5();
	}

    glutPostRedisplay();
}

void key1()
{
	dynamicModel(dataPath, b['o'], false);
	glutPostRedisplay();
}

void key2()
{
	checkModel();
}

bool loadVtx(char *cfile, char *ofile, bool orig);
extern void findMatch();

void key6()
{
	//printf("Checking SelfCD GPU-SH...\n");
	//checkSelfGPU_SH();
}

void key4()
{
	//printf("Checking SelfCD CPU...\n");
	//checkSelfCPU_Refit(NULL);
}

void key5()
{
	//printf("Checking SelfCD GPU-BVH...\n");
	//checkSelfGPU();
}

void key3()
{
#if 0
	//loadVtx("E:\\temp2\\cpu-fw\\c.vtx", "E:\\temp2\\cpu-fw\\c.vto", true);
	//loadVtx("E:\\temp2\\cpu-fw\\b.vtx", "E:\\temp2\\cpu-fw\\b.vto", false);
	printf("Checking SelfCD CPU Naive ...\n");
	//checkCCD();
	
	checkSelfCPU_Naive();

	//checkSelfIJ(267106, 593797);
	//checkSelfIJ(593798, 593797);
#endif
}

void keyQ()
{
}

void key9()
{
}

void quit()
{
	quitModel();
	exit(0);
}

void printLight()
{
	printf("Light: %f, %f, %f, %f\n", lightpos[0], lightpos[1], lightpos[2], lightpos[3]);
}

void updateLight()
{
	glLightfv(GL_LIGHT0, GL_POSITION, &lightpos[0]);
}

void endCapture()
{
}

void key_self_cd(unsigned char k, int x, int y)
{
	b[k] = !b[k];

	switch (k) {
	case 27:
	case 'q':
		quit();
		break;
	}

	object.keyboard(k, x, y);
	glutPostRedisplay();
}

void key(unsigned char k, int x, int y)
{
	b[k] = ! b[k];

    switch(k) {
    case 27:
    case 'q':
		quit();
        break;
		
	case 'x':
		{
			if (b['x'])
				printf("Starting screen capturing.\n");
			else
				printf("Ending screen capturing.\n");

			break;
		}

	// adjust light source
	case 'L':
		lightpos[0] += 0.2f;
		updateLight();
		break;

	case 'J':
		lightpos[0] -= 0.2f;
		updateLight();
		break;

	case 'I':
		lightpos[1] += 0.2f;
		updateLight();
		break;

	case 'K':
		lightpos[1] -= 0.2f;
		updateLight();
		break;

	case 'O':
		lightpos[2] += 0.2f;
		updateLight();
		break;

	case 'U':
		lightpos[2] -= 0.2f;
		updateLight();
		break;

	case 'r':
		initModel(0, NULL, dataPath, stFrame);
		break;

	case '1':
		key1();
		break;

	case '2':
		key2();
		break;

	case '=':
		level++;
		break;

	case '-':
		level--;
		break;

	case '3':
		key3();
		//key2();
		break;

	case '4':
		key4();
		break;

	case '5':
		key5();
		break;

	case '6':
		key6();
		break;

	case '9':
		key9();
		break;

	case '?':
		keyQ();
		break;
	}

    object.keyboard(k, x, y);    
	glutPostRedisplay();
}

void resize(int w, int h)
{
    if (h == 0) h = 1;

    glViewport(0, 0, w, h);
    
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)w/(GLfloat)h, 0.1, 500.0);
    
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    object.reshape(w, h);

    win_w = w; win_h = h;
}

void mouse(int button, int state, int x, int y)
{
    object.mouse(button, state, x, y);
}

void motion(int x, int y)
{
    object.motion(x, y);
}

void main_menu(int i)
{
    key((unsigned char) i, 0, 0);
}

void initMenu()
{    
    glutCreateMenu(main_menu);
	glutAddMenuEntry("Toggle animation [d]", 'd');
	glutAddMenuEntry("Toggle obb/aabb [o]", 'o');
	glutAddMenuEntry("========================", '=');
	glutAddMenuEntry("Toggle rebuild/refit  (aabb) [r]", 'r');
	glutAddMenuEntry("Increasing boxes level(aabb) [=]", '=');
	glutAddMenuEntry("Decreasing boxes level(aabb) [-]", '-');
	glutAddMenuEntry("========================", '=');
    glutAddMenuEntry("Toggle wireframe [w]", 'w');
	glutAddMenuEntry("Toggle lighting [l]", 'l');
	glutAddMenuEntry("Toggle avi recording [x]", 'x');
	glutAddMenuEntry("Save camera[s]", 's');
	glutAddMenuEntry("Reset camera[t]", 't');
	glutAddMenuEntry("========================", '=');
    glutAddMenuEntry("Quit/q [esc]", '\033');
    glutAttachMenu(GLUT_RIGHT_BUTTON);
}

void usage_self_cd()
{
	printf("Keys:\n");
	//printf("c - Checking self-collisions.\n");
	printf("t - Toggle display colliding/all triangles.\n");
	printf("q/ESC - Quit.\n\n");
	printf("Mouse:\n");
	printf("Left Btn - Obit.\n");
	printf("Ctrl+Left Btn - Zoom.\n");
	printf("Shift+Left Btn - Pan.\n");
}

void usage()
{
	printf("Keys:\n");
	printf("1 - next frame\n");
	printf("2 - check collisions between the cloth and the body\n");
	printf("3 - checking self-collisions of the cloth, CPU naively.\n");
	printf("4 - checking self-collisions of the cloth, CPU with bvh.\n");
	printf("5 - checking self-collisions of the cloth, GPU with bvh.\n");
	printf("6 - find the closest vid of the input cloth vid (for handle).\n");
}

//#define NO_UI
#ifdef NO_UI

#else


//int main_cd(int argc, char **argv)
int main(int argc, char **argv)
{
	if (argc < 3) {
		printf("usage: %s cloth.obj body.obj\n", argv[0]);
		return -1;
	}

	usage_self_cd();

	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGBA | GLUT_STENCIL);
	glutInitWindowSize(win_w, win_h);
	glutCreateWindow("Cloth Viewer");

	initOpengl();
	//initModel(argc, argv, dataPath, stFrame);
	initModel(argv[1], argv[2]);
	//key2();

	object.configure_buttons(1);
	object.dolly.dolly[2] = -3;
	object.trackball.incr = rotationf(vec3f(1, 1, 0), 0.05);

	glutDisplayFunc(display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutIdleFunc(idle);
	//glutKeyboardFunc(key_self_cd);
	glutKeyboardFunc(key);
	glutReshapeFunc(resize);

	initMenu();

	initSetting();

	glutMainLoop();

	quit();
	return 0;
}

#endif

void CaptureScreen(int Width, int Height)
{
#ifdef WIN32
	static int captures=0;
	char filename[20];

	sprintf( filename, "Data/%04d.bmp", captures );
	captures++;

	BITMAPFILEHEADER bf;
	BITMAPINFOHEADER bi;

	char *image = new char[Width*Height*3];
	FILE *file = fopen( filename, "wb");

	if( image!=NULL )
	{
		if( file!=NULL ) 
		{
			glReadPixels( 0, 0, Width, Height, GL_BGR_EXT, GL_UNSIGNED_BYTE, image );

			memset( &bf, 0, sizeof( bf ) );
			memset( &bi, 0, sizeof( bi ) );

			bf.bfType = 'MB';
			bf.bfSize = sizeof(bf)+sizeof(bi)+Width*Height*3;
			bf.bfOffBits = sizeof(bf)+sizeof(bi);
			bi.biSize = sizeof(bi);
			bi.biWidth = Width;
			bi.biHeight = Height;
			bi.biPlanes = 1;
			bi.biBitCount = 24;
			bi.biSizeImage = Width*Height*3;

			fwrite( &bf, sizeof(bf), 1, file );
			fwrite( &bi, sizeof(bi), 1, file );
			fwrite( image, sizeof(unsigned char), Height*Width*3, file );

			fclose( file );
		}
		delete[] image;
	}
#endif
}
