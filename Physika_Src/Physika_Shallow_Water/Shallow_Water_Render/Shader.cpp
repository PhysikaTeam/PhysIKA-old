#include "Shader.h"
namespace Physika {
Shader::Shader(GLchar const *vertexPath, GLchar const *fragmentPath) {

	std::ifstream vertexShaderFile;
	std::ifstream fragmentShaderFile;

	vertexShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
	fragmentShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

	std::string vShaderSourceString, fShaderSourceString;

	try {
		vertexShaderFile.open(vertexPath);
		fragmentShaderFile.open(fragmentPath);

		std::stringstream vertexShaderSourceStream, fragmentShaderSourceStream;

		vertexShaderSourceStream << vertexShaderFile.rdbuf();
		fragmentShaderSourceStream << fragmentShaderFile.rdbuf();

		vShaderSourceString = vertexShaderSourceStream.str();
		fShaderSourceString = fragmentShaderSourceStream.str();
	}
	catch (std::ifstream::failure e) {
		std::cout << "ERROR::SHADER::FILE_READ_FAILURE" << std::endl;
	}

	const GLchar* vShaderCode = vShaderSourceString.c_str();
	const GLchar* fShaderCode = fShaderSourceString.c_str();

	GLuint vertexObj, fragmentObj;
	GLint success;
	GLchar infoLog[512];

	vertexObj = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vertexObj, 1, &vShaderCode, NULL);
	glCompileShader(vertexObj);
	glGetShaderiv(vertexObj, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(vertexObj, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	fragmentObj = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fragmentObj, 1, &fShaderCode, NULL);
	glCompileShader(fragmentObj);
	glGetShaderiv(fragmentObj, GL_COMPILE_STATUS, &success);
	if (!success) {
		glGetShaderInfoLog(fragmentObj, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
	}

	program = glCreateProgram();
	glAttachShader(program, vertexObj);
	glAttachShader(program, fragmentObj);
	glLinkProgram(program);

	glGetProgramiv(program, GL_LINK_STATUS, &success);
	if (!success) {
		glGetProgramInfoLog(program, 512, NULL, infoLog);
		std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
	}

	glDeleteShader(vertexObj);
	glDeleteShader(fragmentObj);
}

void Shader::deleteProgram() {
	glDeleteProgram(program);
}

void Shader::use() {
	glUseProgram(program);
}

GLuint Shader::getProgram() {
	return program;
}
}
