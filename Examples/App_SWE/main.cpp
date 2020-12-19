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

#include "IO\Image_IO\image.h"
#include "IO\Image_IO\png_io.h"
#include "IO\Image_IO\image_io.h"

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

void CreateScene()
{
	SceneGraph& scene = SceneGraph::getInstance();
	scene.setUpperBound(Vector3f(1.5, 1, 1.5));
	scene.setLowerBound(Vector3f(-0.5, 0, -0.5));

	std::shared_ptr<HeightFieldNode<DataType3f>> root = scene.createNewScene<HeightFieldNode<DataType3f>>();

	auto ptRender = std::make_shared<HeightFieldRenderModule>();
	ptRender->setColor(Vector3f(1, 0, 0));
	root->addVisualModule(ptRender);

	root->loadParticles(Vector3f(0, 0, 0), Vector3f(2, 1.5, 2), 512, 0.2, 0.998);
	//root->loadParticlesFromImage(Vector3f(0, 0, 0), Vector3f(2, 1, 2), 512, 0, 0.998);
	//root->loadParticlesFromImage(Vector3f(0, 0.5, 0), Vector3f(1, 1, 1), 1024, 0.2, 0.998);

	root->setMass(100);


	//std::shared_ptr<RigidBody<DataType3f>> rigidbody = std::make_shared<RigidBody<DataType3f>>();
	//root->addRigidBody(rigidbody);
	//rigidbody->loadShape("../../Media/bowl/bowl.obj");
	//rigidbody->setActive(false);
}

int main()
{
	CreateScene();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	Log::sendMessage(Log::Info, "Simulation begin");

	GLApp window;
	window.createWindow(1024, 768);

	window.mainLoop();

	Log::sendMessage(Log::Info, "Simulation end!");
			//初始化输出测试

	/*读取文件进行压缩*/
	//Image *image = new Image();
	////std::string filename1 = "F:\\新建文件夹\\大四第一学期\\swe\\river.png";
	//std::string filename1 = "F:\\新建文件夹\\大四第一学期\\swe\\river.png";
	//ImageIO::load(filename1, image);

	//
	//image->mergeImage(8,8);
	//string filename2 = "..\\..\\..\\Examples\\App_SWE\\river8-8.png";
	//ImageIO::save(filename2, image);
	return 0;
}