#include <iostream>
#include <memory>
#include <cuda_runtime_api.h>
#include <GL/freeglut.h>

#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"
#include "Physika_Geometry/Boundary_Meshes/surface_mesh.h"
#include "Physika_GUI/Glut_Window/glut_window.h"

#include "Physika_Render/ColorBar/ColorMap/color_map.h"

#include "Physika_Render/Lights/directional_light.h"
#include "Physika_Render/Lights/point_light.h"
#include "Physika_Render/Lights/spot_light.h"
#include "Physika_Render/Lights/flash_light.h"

#include "Physika_Render/Render_Scene_Config/render_scene_config.h"
#include "Physika_Render/Point_Render/point_render_util.h"
#include "Physika_Render/Point_Render/point_render_task.h"
#include "Physika_Render/Point_Render/point_gl_cuda_buffer.h"
#include "Physika_Render/Point_Render/point_vector_render_task.h"
#include "Physika_Render/Utilities/gl_cuda_buffer_test_tool.h"

#include "Physika_Framework/Framework/SceneGraph.h"
#include "Physika_Dynamics/ParticleSystem/ParticleSystem.h"
#include "Physika_Framework/Topology/PointSet.h"
#include "Physika_Framework/Framework/Log.h"
#include "Physika_Render/VisualModule/PointRenderModule.h"

using namespace std;
using namespace Physika;

SurfaceMesh<double> mesh;
std::shared_ptr<PointRenderUtil> point_render_util;
PointSet<Vector3f>* pSet;

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

void initFunction()
{
	//---------------------------------------------------------------------------------------------------------------------
	RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
// 	point_render_util = make_shared<PointRenderUtil>();
// 
// 	DeviceArray<float3>* xyz = (DeviceArray<float3>*)pSet->getPoints();
// 	PointGLCudaBuffer point_gl_cuda_buffer = point_render_util->mapPointGLCudaBuffer(xyz->size());
// 	
// 	point_render_util->unmapPointGLCudaBuffer();
// 
// 	//---------------------------------------------------------------------------------------------------------------------
// 	auto point_render_task = make_shared<PointRenderTask>(point_render_util);
// 	//point_render_task->disableUsePointSprite();
// 	point_render_task->setPointScaleForPointSprite(3.0);
// 	render_scene_config.pushBackRenderTask(point_render_task);

	auto flash_light = make_shared<FlashLight>();
	//flash_light->setAmbient(Color4f::White());
	render_scene_config.pushBackLight(flash_light);

	render_scene_config.zoomCameraOut(3.0);
	render_scene_config.translateCameraRight(0.5);


	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glClearDepth(1.0);
	glClearColor(0.49, 0.49, 0.49, 1.0);

	glEnable(GL_DEPTH_TEST);
	//glEnable(GL_CULL_FACE);
}

void displayFunction()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(1.0, 1.0);

	RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
	render_scene_config.renderAllTasks();

	GlutWindow * cur_window = (GlutWindow*)glutGetWindowData();
	cur_window->displayFrameRate();
// 	DeviceArray<float3>* xyz = (DeviceArray<float3>*)pSet->getPoints();
// 	PointGLCudaBuffer point_gl_cuda_buffer = point_render_util->mapPointGLCudaBuffer(xyz->size());
// 	cudaMemcpy(point_gl_cuda_buffer.getCudaPosPtr(), xyz->getDataPtr(), sizeof(float) * 3 * xyz->size(), cudaMemcpyDeviceToDevice);
// 	point_render_util->unmapPointGLCudaBuffer();
	cur_window->getScene()->takeOneFrame();
	cur_window->getScene()->draw();
	glutPostRedisplay();
	glutSwapBuffers();
}

void keyboardFunction(unsigned char key, int x, int y)
{
	GlutWindow::bindDefaultKeys(key, x, y);
	switch (key)
	{
	case 't':
		cout << "test\n";
		break;
	default:
		break;
	}
}

int main()
{
	GlutWindow glut_window;
	cout << "Window name: " << glut_window.name() << "\n";
	cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";

	RenderSceneConfig & render_scene_config1 = RenderSceneConfig::getSingleton();

	Log::setOutput("console_log.txt");
	Log::setLevel(Log::Info);
	Log::setUserReceiver(&RecieveLogMessage);
	
	Log::sendMessage(Log::Info, "Initialization begin");
	std::shared_ptr<SceneGraph> scene = SceneGraph::getInstance();

	std::shared_ptr<Node> root = scene->createNewScene<Node>("root");

	pSet = new PointSet<Vector3f>();
	std::vector<Vector3f> positions;
	for (float x = 0.4; x < 0.6; x += 0.005f) {
		for (float y = 0.1; y < 0.2; y += 0.005f) {
			for (float z = 0.4; z < 0.6; z += 0.005f) {
				positions.push_back(Vector3f(float(x), float(y), float(z)));
			}
		}
	}
	pSet->initialize(positions);

	root->setTopologyModule(pSet);

	auto pS2 = new ParticleSystem<DataType3f>();
	root->setNumericalModel(pS2);
	pS2->initialize();

	auto render = new PointRenderModule();
	render->setParent(root.get());
	render->initialize();
	root->addVisualModule(render);


	glut_window.setScene(scene);

	RenderSceneConfig & render_scene_config = RenderSceneConfig::getSingleton();
	//render_scene_config.setCameraPosition(Vector<double, 3>(0, 0, 200));
	//render_scene_config.setCameraFocusPosition(Vector<double, 3>(0, 0, 0));
	render_scene_config.setCameraNearClip(0.1);
	render_scene_config.setCameraFarClip(1.0e3);

	glut_window.setDisplayFunction(displayFunction);
	glut_window.setInitFunction(initFunction);

	cout << "Test GlutWindow with custom display function:\n";
	glut_window.createWindow();
	cout << "Window size: " << glut_window.width() << "x" << glut_window.height() << "\n";
	cout << "Test window with GLUI controls:\n";

	return 0;
}