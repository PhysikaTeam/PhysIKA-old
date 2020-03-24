#include "PyGlutGUI.h"

#include "GUI/GlutGUI/GLApp.h"

void pybind_glut_gui(py::module& m)
{
	py::class_<PhysIKA::GLApp>(m, "GLApp")
		.def(py::init())
		.def("create_window", &PhysIKA::GLApp::createWindow)
		.def("main_loop", &PhysIKA::GLApp::mainLoop)
		.def("name", &PhysIKA::GLApp::name)
		.def("get_width", &PhysIKA::GLApp::getWidth)
		.def("get_height", &PhysIKA::GLApp::getHeight)
		.def("save_screen", (bool (PhysIKA::GLApp::*)()) &PhysIKA::GLApp::saveScreen)
		.def("save_screen", (bool (PhysIKA::GLApp::*)(const std::string &) const) &PhysIKA::GLApp::saveScreen);
}
