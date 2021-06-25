#ifndef _PROGRAM_H_
#define _PROGRAM_H_

class Program {
public:
	int program;

	int vertexShader;

	int fragmentShader;

	int geometryShader;

	Program(int files, char** fileNames, char* options = NULL);

	~Program();
};

#endif
