#include <iostream>
#include <memory>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "GUI/GlutGUI/GLApp.h"

#include "Framework/Framework/SceneGraph.h"
#include "Framework/Framework/Log.h"

#include "Dynamics/RigidBody/RigidBody.h"
#include "Dynamics/HeightField/HeightFieldNode.h"

#include "Rendering/HeightFieldRender.h"

#include "IO/Image_IO/image.h"
#include "IO/Image_IO/png_io.h"
#include "IO/Image_IO/image_io.h"

using namespace std;
using namespace PhysIKA;


void RecieveLogMessage(const Log::Message& m)
{
	switch (m.type)
	{
	case Log::Info:
		cout << ">>>: " << m.text << endl; break;
	case Log::Warning:
		cout << "???: " << m.text << endl; break; 
	case Log::Error:
		cout << "!!!: " << m.text << endl; break;
	case Log::User:
		cout << ">>>: " << m.text << endl; break;
	default: break;
	}
}

//mode: choose which scene to create. mode=1 creates the basic scene, otherwise creates city scene.
void CreateScene(int mode = 1)
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.5, 1, 1.5));
	scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

	std::shared_ptr<HeightFieldNode<DataType3f>> root = scene.createNewScene<HeightFieldNode<DataType3f>>();
	

	auto ptRender = std::make_shared<HeightFieldRenderModule>();
	ptRender->setColor(Vector3f(1, 0, 0));
	root->addVisualModule(ptRender);

	if(mode == 1)
		root->loadParticles(Vector3f(0, 0, 0), Vector3f(2, 1.5, 2), 512, 0.7, 1);
	else
	{
		std::string filename1 = "../../../Examples/App_SWE/terrain4-4.png";//The pixel count is 1024*1024
		std::string filename2 = "../../../Examples/App_SWE/river4-4.png";
		root->loadParticlesFromImage(filename1, filename2, 0.1, 1);
	}
	root->setMass(100);

	//root->run(1,0.03);
	//auto result = root->outputSolid();
	//std::cout << result[0];

}

void executeOnce() 
{
	std::shared_ptr<HeightFieldNode<DataType3f>> root(new HeightFieldNode<DataType3f>);

	std::string filename1 = "../../../Examples/App_SWE/terrain4-4.png";//The pixel count is 1024*1024
	std::string filename2 = "../../../Examples/App_SWE/river4-4.png";

	root->loadParticlesFromImage(filename1, filename2, 0.1, 0.998);

	float dt = 1.0 / 60;
	std::vector<Real> vec1 = root->outputDepth();
	root->run(1, dt);
	std::vector<Real> vec2 = root->outputDepth();
	std::cout << "the depth difference:" << std::endl;
	for (int i = 0; i < vec1.size(); i++) 
	{
		if (vec1[i] != vec2[i]) 
		{
			std::cout << i << std::endl;
		}
	}
}

int main()
{
#if 0
	executeOnce();
#else
	CreateScene(1);

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
#endif

	return 0;
}