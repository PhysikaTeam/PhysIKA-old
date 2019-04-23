pragma once
#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

//#define GLEW_STATIC
#define GLEW_NO_GLU

#include <GL/glew.h>

class Shader {
public:

	Shader(GLchar const * vertexPath, GLchar const * fragmentPath);
	//~Shader();

	void deleteProgram();
	void use();

	GLuint getProgram();

private:
	GLuint program;
};
