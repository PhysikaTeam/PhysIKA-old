#include "render.h"

GLInterface::GLInterface(int window_width, int window_height) :
	window(nullptr),
	aspect_ratio(GLfloat(window_width) / (GLfloat)window_height)
{

	if (!glfw_initialized) {
		glfw_initialized = glfwInit();
		if (glfw_initialized != GL_TRUE) {
			std::cout << "Error: unable to initialize GLFW\n" << std::endl;
			return;
		}
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

	window = glfwCreateWindow(window_width, window_height, "Shallow water simulation", NULL, NULL);
	if (!window) {
		std::cout << "Error: unable to create a window, check window hints\n" << std::endl;
		glfwTerminate();
		return;
	}

	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK) {
		std::cout << "Error: unable to initialize GLEW\n" << std::endl;
		glfwTerminate();
		return;
	}

	int pxl_width, pxl_height;
	glfwGetFramebufferSize(window, &pxl_width, &pxl_height);
	glViewport(0, 0, pxl_width, pxl_height);

	glEnable(GL_DEPTH_TEST);    
	glfwSetKeyCallback(window, ViewUtils::key_callback);
	glfwSetCursorPosCallback(window, ViewUtils::mouse_callback);
	glfwSetScrollCallback(window, ViewUtils::scroll_callback);

	glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);    
}

VisualEngine::VisualEngine(size_t x_cells, size_t y_cells, float dx, int window_width, int window_height) :

	GLInterface(window_width, window_height),

	waterShaderProgram("shaders/waterVertexShader.txt", "shaders/waterFragmentShader.txt"),
	surfaceShaderProgram("shaders/surfaceVertexShader.txt", "shaders/surfaceFragmentShader.txt"),
	mesh_rows(x_cells),
	mesh_columns(y_cells)
{
	std::vector<GLfloat> mesh_xy(2 * x_cells * y_cells);
	for (size_t x = 0, linear_index = 0; x < x_cells; ++x) {
		for (size_t y = 0; y < y_cells; ++y) {
			mesh_xy[linear_index++] = x * dx;
			mesh_xy[linear_index++] = y * dx;
		}
	}

	std::vector<GLuint> mesh_indices(2 * (x_cells - 1) * y_cells);
	for (size_t x = 0, vertex_number = 0; x < x_cells - 1; ++x) {
		for (size_t y = 0; y < y_cells; ++y) {
			mesh_indices[vertex_number++] = x * y_cells + y;
			mesh_indices[vertex_number++] = (x + 1) * y_cells + y;
		}
	}

	glGenBuffers(1, &mesh_xyVBO);
	glGenBuffers(1, &meshEBO);

	glGenVertexArrays(1, &surfaceVAO);
	glBindVertexArray(surfaceVAO);
	glBindBuffer(GL_ARRAY_BUFFER, mesh_xyVBO);
	glBufferData(GL_ARRAY_BUFFER, mesh_xy.size() * sizeof(GLfloat), mesh_xy.data(), GL_STATIC_DRAW);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, mesh_indices.size() * sizeof(GLuint), mesh_indices.data(), GL_STATIC_DRAW);
	glBindVertexArray(0);


	glGenVertexArrays(1, &waterVAO);
	glBindVertexArray(waterVAO);

	glBindBuffer(GL_ARRAY_BUFFER, mesh_xyVBO);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, meshEBO);
	glBindVertexArray(0);

	glClearColor(0.3f, 0.3f, 0.4f, 0.3f);
}

VisualEngine::~VisualEngine() {
	surfaceShaderProgram.deleteProgram();
	waterShaderProgram.deleteProgram();
	glDeleteVertexArrays(1, &waterVAO);
	glDeleteVertexArrays(1, &surfaceVAO);
	glDeleteBuffers(1, &waterHeightVBO);
	glDeleteBuffers(1, &surfaceLevelVBO);
	glDeleteBuffers(1, &mesh_xyVBO);
	glDeleteBuffers(1, &meshEBO);
}

void VisualEngine::update_vertex_values(std::vector<GLfloat> *water_height, std::vector<GLfloat> *surface_level) {
	static bool first_time = true;
	if (surface_level && first_time) {
		glGenBuffers(1, &surfaceLevelVBO);
		glBindVertexArray(surfaceVAO);
		glBindBuffer(GL_ARRAY_BUFFER, surfaceLevelVBO);
		glBufferData(GL_ARRAY_BUFFER, surface_level->size() * sizeof(GLfloat), surface_level->data(), GL_STATIC_DRAW);
		glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (GLvoid*)0);
		glEnableVertexAttribArray(1);
		glBindVertexArray(0);
	}
	if (water_height) {
		if (first_time) {
			glGenBuffers(1, &waterHeightVBO);
		}
		glBindVertexArray(waterVAO);
		glBindBuffer(GL_ARRAY_BUFFER, waterHeightVBO);
		if (first_time) {
			glBufferData(GL_ARRAY_BUFFER, water_height->size() * sizeof(GLfloat), water_height->data(), GL_STREAM_DRAW);
			glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(GLfloat), (GLvoid*)0);
			first_time = false;
			glEnableVertexAttribArray(1);
		}
		else {
			glBufferSubData(GL_ARRAY_BUFFER, 0, water_height->size() * sizeof(GLfloat), water_height->data());
		}
		glBindVertexArray(0);
	}
}

void VisualEngine::render() {
	glfwPollEvents();

	static GLfloat lastFrame = glfwGetTime();
	GLfloat currentTime = glfwGetTime();
	ViewUtils::deltaTime = currentTime - lastFrame;
	lastFrame = currentTime;

	ViewUtils::do_camera_movement();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glm::mat4 view = ViewUtils::camera.GetViewMatrix();
	glm::mat4 projection = glm::perspective(glm::radians(ViewUtils::camera.Zoom), aspect_ratio, 0.1f, 100.0f);

	surfaceShaderProgram.use();
	glUniformMatrix4fv(glGetUniformLocation(surfaceShaderProgram.getProgram(), "view"), 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(glGetUniformLocation(surfaceShaderProgram.getProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(projection));

	glBindVertexArray(surfaceVAO);
	for (GLuint row = 0; row < mesh_rows - 1; ++row) {
		glDrawElements(GL_TRIANGLE_STRIP, mesh_columns * 2, GL_UNSIGNED_INT, (GLvoid*)(row * mesh_columns * 2 * sizeof(GLuint)));
	}
	glBindVertexArray(0);

	waterShaderProgram.use();
	glUniformMatrix4fv(glGetUniformLocation(waterShaderProgram.getProgram(), "view"), 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(glGetUniformLocation(waterShaderProgram.getProgram(), "projection"), 1, GL_FALSE, glm::value_ptr(projection));
	glBindVertexArray(waterVAO);
	for (GLuint row = 0; row < mesh_rows - 1; ++row) {
		glDrawElements(GL_TRIANGLE_STRIP, mesh_columns * 2, GL_UNSIGNED_INT, (GLvoid*)(row * mesh_columns * 2 * sizeof(GLuint)));
	}
	glBindVertexArray(0);

	glfwSwapBuffers(window);
}

bool VisualEngine::should_stop() {
	return (bool)glfwWindowShouldClose(window);
}

bool VisualEngine::start_simulation() {
	return ViewUtils::keys[GLFW_KEY_SPACE];
}

GLint GLInterface::glfw_initialized = 0;

GLInterface::~GLInterface() {
	if (glfw_initialized) {
		glfwTerminate();
		glfw_initialized = false;
	}

}

namespace ViewUtils {

	Camera camera(glm::vec3(5.0f, 4.0f, -7.0f), glm::vec3(0.0f, 1.0f, 0.0f), 90.0f);

	bool keys[1024];
	GLfloat deltaTime = 0.0;

	void do_camera_movement() {
		if (keys[GLFW_KEY_W]) {
			camera.ProcessKeyboard(FORWARD, deltaTime);
		}
		if (keys[GLFW_KEY_S]) {
			camera.ProcessKeyboard(BACKWARD, deltaTime);
		}
		if (keys[GLFW_KEY_A]) {
			camera.ProcessKeyboard(LEFT, deltaTime);
		}
		if (keys[GLFW_KEY_D]) {
			camera.ProcessKeyboard(RIGHT, deltaTime);
		}
	}

	void scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
		camera.ProcessMouseScroll(yoffset);
	}

	void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode) {

		if (key == GLFW_KEY_ESCAPE && action == GLFW_RELEASE) {
			glfwSetWindowShouldClose(window, GL_TRUE);
			return;
		}

		if (key >= 0 && key < 1024) {
			if (action == GLFW_PRESS) {
				keys[key] = true;
			}
			else if (action == GLFW_RELEASE) {
				keys[key] = false;
			}
		}
	}

	void mouse_callback(GLFWwindow *window, double xpos, double ypos) {

		static GLfloat lastX = 400.0f, lastY = 300.0f;
		static bool firstMouseMove = true;
		if (firstMouseMove) {
			lastX = (GLfloat)xpos;
			lastY = (GLfloat)ypos;
			firstMouseMove = false;
		}

		GLfloat xoffset = xpos - lastX;
		GLfloat yoffset = lastY - ypos; 
		lastX = xpos;
		lastY = ypos;

		camera.ProcessMouseMovement(xoffset, yoffset);
	}
}
