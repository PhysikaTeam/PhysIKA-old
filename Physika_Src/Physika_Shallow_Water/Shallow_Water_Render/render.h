#pragma once
//#define GLEW_STATIC
#define GLEW_NO_GLU

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/gtc/type_ptr.hpp>

//#include "SOIL.h"

#include <omp.h>
#include <vector>
#include <iostream>

#include "Shader.h"
#include "Physika_Shallow_Water/Shallow_Water_GUI/Camera.h"

class GLInterface {
public:
	GLInterface(int window_width, int window_height);
	virtual ~GLInterface();

protected:
	GLFWwindow* window;
	GLfloat aspect_ratio;
	static GLint glfw_initialized;
};


class VisualEngine : public GLInterface {
public:
	VisualEngine(size_t x_cells, size_t y_cells, float dx, int window_width = 800, int window_height = 600);
	~VisualEngine();

	void update_vertex_values(std::vector<GLfloat> *water_height = nullptr, std::vector<GLfloat> *surface_level = nullptr);
	bool start_simulation();
	void render();
	bool should_stop();



private:

	GLuint waterVAO;
	GLuint surfaceVAO;

	GLuint waterHeightVBO;   
	GLuint surfaceLevelVBO;

	GLuint mesh_xyVBO;
	GLuint meshEBO;

	GLuint water_texture;

	std::vector<GLfloat> texCoordinates;
	std::vector<GLuint> meshElements;

	Shader waterShaderProgram;
	Shader surfaceShaderProgram;

	GLuint mesh_rows;
	GLuint mesh_columns;

	
};

namespace ViewUtils {

	extern Camera camera;

	extern bool keys[1024];
	extern GLfloat deltaTime;

	void key_callback(GLFWwindow* window, int key, int scancode, int action, int mode);
	void mouse_callback(GLFWwindow* window, double xpos, double ypos);
	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);

	void do_camera_movement();
}
