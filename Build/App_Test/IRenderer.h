#ifndef MFD_RENDERER_H
#define MFD_RENDERER_H

#include <GL/glew.h>
#if defined (_WIN32)
#include <GL/wglew.h>
#endif
#include <GL/freeglut.h>
#include "Color.h"

class Camera;

typedef unsigned int uint;

class IRenderer {
public:
	virtual void Render(const Camera &camera) {};
};

#endif

