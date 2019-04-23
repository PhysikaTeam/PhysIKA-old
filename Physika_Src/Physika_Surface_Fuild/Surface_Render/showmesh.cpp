#include "showmesh.h"
#include "stdio.h"
#include "math.h"
#include "GL/glut.h"
#include "boundrecorder.h"
#include "simulator.h"

static const GLfloat afAmbientWhite [] = {0.25, 0.25, 0.25, 1.00};
static const GLfloat afAmbientRed   [] = {0.25, 0.00, 0.00, 1.00};
static const GLfloat afAmbientGreen [] = {0.00, 0.25, 0.00, 1.00};
static const GLfloat afAmbientBlue  [] = {0.00, 0.00, 0.25, 1.00};
static const GLfloat afDiffuseWhite [] = {0.75, 0.75, 0.75, 1.00};
static const GLfloat afDiffuseRed   [] = {0.75, 0.00, 0.00, 1.00};
static const GLfloat afDiffuseGreen [] = {0.00, 0.75, 0.00, 1.00};
static const GLfloat afDiffuseBlue  [] = {0.00, 0.00, 0.75, 1.00};
static const GLfloat afSpecularWhite[] = {1.00, 1.00, 1.00, 1.00};
static const GLfloat afSpecularRed  [] = {1.00, 0.25, 0.25, 1.00};
static const GLfloat afSpecularGreen[] = {0.25, 1.00, 0.25, 1.00};
static const GLfloat afSpecularBlue [] = {0.25, 0.25, 1.00, 1.00};


static GLenum    ePolygonMode = GL_FILL;
static GLboolean bSpin = false;
static GLboolean bLight = true;

static void vIdle();
static void vDrawScene(); 
static void vResize(GLsizei, GLsizei);
static void vKeyboard(unsigned char cKey, int iX, int iY);
static void vSpecial(int iKey, int iX, int iY);

static GLvoid vPrintHelp();
static GLvoid vDrawMeshFaces();

static MyMesh _mesh;
extern Simulator sim;
int swframe;

void show_mesh ( int argc, char *argv[] )
{
	swframe = 0;
	sim.init ( argc, argv );

	GLfloat afPropertiesAmbient[] = { 0.50, 0.50, 0.50, 1.00 };
	GLfloat afPropertiesDiffuse[] = { 0.75, 0.75, 0.75, 1.00 };
	GLfloat afPropertiesSpecular[] = { 1.00, 1.00, 1.00, 1.00 };

	GLsizei iWidth = 800.0;
	GLsizei iHeight = 600.0;

	glutInit ( &argc, argv );
	glutInitWindowPosition ( 0, 0 );
	glutInitWindowSize ( iWidth, iHeight );
	glutInitDisplayMode ( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );
	glutCreateWindow ( "OpenGL" );
	glutDisplayFunc ( vDrawScene );
	glutIdleFunc ( vIdle );
	glutReshapeFunc ( vResize );
	glutKeyboardFunc ( vKeyboard );
	glutSpecialFunc ( vSpecial );

	glClearColor ( 0.0, 0.0, 0.0, 1.0 );
	glClearDepth ( 1.0 );

	glEnable ( GL_DEPTH_TEST );
	glEnable ( GL_LIGHTING );
	glPolygonMode ( GL_FRONT_AND_BACK, ePolygonMode );

	glLightfv ( GL_LIGHT0, GL_AMBIENT, afPropertiesAmbient );
	glLightfv ( GL_LIGHT0, GL_DIFFUSE, afPropertiesDiffuse );
	glLightfv ( GL_LIGHT0, GL_SPECULAR, afPropertiesSpecular );
	glLightModelf ( GL_LIGHT_MODEL_TWO_SIDE, 1.0 );

	glEnable ( GL_LIGHT0 );

	glMaterialfv ( GL_BACK, GL_AMBIENT, afAmbientGreen );
	glMaterialfv ( GL_BACK, GL_DIFFUSE, afDiffuseGreen );
	glMaterialfv ( GL_FRONT, GL_AMBIENT, afAmbientBlue );
	glMaterialfv ( GL_FRONT, GL_DIFFUSE, afDiffuseBlue );
	glMaterialfv ( GL_FRONT, GL_SPECULAR, afSpecularWhite );
	glMaterialf ( GL_FRONT, GL_SHININESS, 25.0 );

	vResize ( iWidth, iHeight );

	vPrintHelp ();
	glutMainLoop ();

	sim.clear ();
}

void show_mesh(int argc, char *argv[], MyMesh const &mesh)
{ 
	_mesh = mesh;

	GLfloat afPropertiesAmbient [] = {0.50, 0.50, 0.50, 1.00}; 
	GLfloat afPropertiesDiffuse [] = {0.75, 0.75, 0.75, 1.00}; 
	GLfloat afPropertiesSpecular[] = {1.00, 1.00, 1.00, 1.00}; 

	GLsizei iWidth = 800.0; 
	GLsizei iHeight = 600.0; 

	glutInit(&argc, argv);
	glutInitWindowPosition( 0, 0);
	glutInitWindowSize(iWidth, iHeight);
	glutInitDisplayMode( GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE );
	glutCreateWindow( "OpenGL" );
	glutDisplayFunc( vDrawScene );
	glutIdleFunc( vIdle );
	glutReshapeFunc( vResize );
	glutKeyboardFunc( vKeyboard );
	glutSpecialFunc( vSpecial );

	glClearColor( 0.0, 0.0, 0.0, 1.0 ); 
	glClearDepth( 1.0 ); 

	glEnable(GL_DEPTH_TEST); 
	glEnable(GL_LIGHTING);
	glPolygonMode(GL_FRONT_AND_BACK, ePolygonMode);

	glLightfv( GL_LIGHT0, GL_AMBIENT,  afPropertiesAmbient); 
	glLightfv( GL_LIGHT0, GL_DIFFUSE,  afPropertiesDiffuse); 
	glLightfv( GL_LIGHT0, GL_SPECULAR, afPropertiesSpecular); 
	glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0); 

	glEnable( GL_LIGHT0 ); 

	glMaterialfv(GL_BACK,  GL_AMBIENT,   afAmbientGreen); 
	glMaterialfv(GL_BACK,  GL_DIFFUSE,   afDiffuseGreen); 
	glMaterialfv(GL_FRONT, GL_AMBIENT,   afAmbientBlue); 
	glMaterialfv(GL_FRONT, GL_DIFFUSE,   afDiffuseBlue); 
	glMaterialfv(GL_FRONT, GL_SPECULAR,  afSpecularWhite); 
	glMaterialf( GL_FRONT, GL_SHININESS, 25.0); 

	vResize(iWidth, iHeight); 

	vPrintHelp();
	glutMainLoop(); 
}

static GLvoid vPrintHelp()
{
	printf("Mesh Viewer\n");
	printf("w  wireframe on/off\n");
	printf("l  toggle lighting / color-by-normal\n");
	printf("Home  spin scene on/off\n");
}


static void vResize( GLsizei iWidth, GLsizei iHeight ) 
{ 
	GLfloat fAspect, fHalfWorldSize = (1.4142135623730950488016887242097/2); 

	glViewport( 0, 0, iWidth, iHeight ); 
	glMatrixMode (GL_PROJECTION);
	glLoadIdentity ();

	if(iWidth <= iHeight)
	{
		fAspect = (GLfloat)iHeight / (GLfloat)iWidth;
		glOrtho(-fHalfWorldSize, fHalfWorldSize, -fHalfWorldSize*fAspect,
			fHalfWorldSize*fAspect, -10*fHalfWorldSize, 10*fHalfWorldSize);
	}
	else
	{
		fAspect = (GLfloat)iWidth / (GLfloat)iHeight; 
		glOrtho(-fHalfWorldSize*fAspect, fHalfWorldSize*fAspect, -fHalfWorldSize,
			fHalfWorldSize, -10*fHalfWorldSize, 10*fHalfWorldSize);
	}

	glMatrixMode( GL_MODELVIEW );
}

static void vKeyboard(unsigned char cKey, int iX, int iY)
{
	switch(cKey)
	{
	case 'w' :
		{
			if(ePolygonMode == GL_LINE)
			{
				ePolygonMode = GL_FILL;
			}
			else
			{
				ePolygonMode = GL_LINE;
			}
			glPolygonMode(GL_FRONT_AND_BACK, ePolygonMode);
		} break;
	case 'l' :
		{
			if(bLight)
			{
				glDisable(GL_LIGHTING);//use vertex colors
			}
			else
			{
				glEnable(GL_LIGHTING);//use lit material color
			}

			bLight = !bLight;
		};
	}
}


static void vSpecial(int iKey, int iX, int iY)
{
	switch(iKey)
	{
	case GLUT_KEY_HOME :
		{
			bSpin = !bSpin;
		} break;
	}
}

static void vIdle()
{
	glutPostRedisplay();
}

static void vDrawScene() 
{
	static GLfloat fPitch = 0.0;
	static GLfloat fYaw   = 0.0;

	glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT ); 

	glPushMatrix(); 

	if(bSpin)
	{
		fPitch += 4.0;
		fYaw   += 2.5;
	}

	glTranslatef( 0.0, -0.0, 0.0 );  
	glRotatef( -fPitch, 1.0, 0.0, 0.0);
	glRotatef(     0.0, 0.0, 1.0, 0.0);
	glRotatef(    fYaw, 0.0, 0.0, 1.0);

	glPushAttrib(GL_LIGHTING_BIT);
	glDisable(GL_LIGHTING);
	glColor3f(1.0, 1.0, 1.0);
	glutWireCube(1.0); 
	glPopAttrib(); 


	glPushMatrix(); 
	glTranslatef(-0.5, -0.5, -0.5);
	glBegin(GL_TRIANGLES);
	vDrawMeshFaces();
	glEnd();
	glPopMatrix(); 

	glPopMatrix(); 

	glutSwapBuffers(); 
}

static GLvoid vDrawMeshFaces()
{
	sim.run_cuda (swframe++);
	//sim.run();
	_mesh = sim.m_mesh;

	MyMesh &mesh(_mesh);
	if (mesh.n_faces() == 0)
		return;
	BoundRecorder<double> xbound, ybound, zbound;
	for (auto v_it = mesh.vertices_begin(); v_it != mesh.vertices_end(); ++v_it) {
		MyMesh::Point const &p(mesh.point(*v_it));
		xbound.insert(p[0]);
		ybound.insert(p[1]);
		zbound.insert(p[2]);
	}
	auto max = [](double a, double b) { return a > b ? a : b; };
	double size = max(max(xbound.get_max() - xbound.get_min(), ybound.get_max() - ybound.get_min()), zbound.get_max() - zbound.get_min());
	double xoff = (size - xbound.get_max() - xbound.get_min()) / 2;
	double yoff = (size - ybound.get_max() - ybound.get_min()) / 2;
	double zoff = (size - zbound.get_max() - zbound.get_min()) / 2;
	mesh.request_face_normals();
	mesh.update_face_normals();

	for (auto f_it = mesh.faces_begin(); f_it != mesh.faces_end(); ++f_it) {
		/*for (auto fv_it = mesh.cfv_ccwiter(*f_it); fv_it.is_valid(); ++fv_it) {
			MyMesh::Point p(mesh.point(*fv_it));
			MyMesh::Normal n(mesh.normal(*f_it));
			MyMesh::Point color = (n + MyMesh::Point(1, 1, 1)) / 2;
			glColor3f(color[0], color[1], color[2]);
			glNormal3f(n[0], n[1], n[2]);
			glVertex3f((p[0] + xoff) / size, (p[1] + yoff) / size, (p[2] + zoff) / size);
		}*/

		for (auto fv_it = mesh.cfv_ccwiter ( *f_it ); fv_it.is_valid (); ++fv_it) {
			MyMesh::VertexHandle vh = *fv_it;
			MyMesh::Point p ( mesh.point ( *fv_it ) );
			glVertex3f ( (p[0] + xoff) / size, (p[1] + yoff) / size, (p[2] + zoff) / size );
		}
	}
	mesh.release_face_normals();
	return;
}
