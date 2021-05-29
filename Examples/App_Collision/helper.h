#ifndef HELPER_H
#define HELPER_H
#include "Dynamics/RigidBody/RigidCollisionBody.h"

using namespace PhysIKA;
extern std::shared_ptr<RigidCollisionBody<DataType3f>> bunny;
void checkCollision();
inline void keyfunc(unsigned char key, int x, int y)
{
	GLApp *window = static_cast<GLApp*>(glutGetWindowData());
	assert(window);
	switch (key)
	{
	case 27: //ESC: close window
		glutLeaveMainLoop();
		return;
	case 's': //s: save screen shot
		window->saveScreen();
		break;
	case 'l':
		bunny->translate({ 0.04, 0, 0 });
		break;
	case 'j':
		bunny->translate({ -0.04, 0, 0 });
		break;
	case 'i':
		bunny->translate({ 0, 0.04, 0 });
		break;
	case 'k':
		bunny->translate({ 0, -0.04, 0 });
		break;
	case 'o':
		bunny->translate({ 0, 0, 0.04 });
		break;
	case 'u':
		bunny->translate({ 0, 0, -0.04 });
		break;
	case 'r':
		bunny->loadSurface("../../Media/bunny/bunny_mesh.obj");
		break;
	}
	checkCollision();
}
#endif