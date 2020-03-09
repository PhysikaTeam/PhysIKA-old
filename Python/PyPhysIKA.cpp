#include <pybind11/pybind11.h>
#include "Core/Utility/CTimer.h"
#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/SceneGraph.h"

namespace py = pybind11;

PYBIND11_MODULE(PyPhysIKA, m) {
	m.doc() = "Glut GUI";
	pybind11::class_<PhysIKA::CTimer>(m, "CTimer")
		.def(pybind11::init())
		.def("start", &PhysIKA::CTimer::start)
		.def("stop", &PhysIKA::CTimer::stop)
		.def("get_elapsed_time", &PhysIKA::CTimer::getElapsedTime);

	pybind11::class_<PhysIKA::GLApp>(m, "GLApp")
		.def(pybind11::init())
		.def("create_window", &PhysIKA::GLApp::createWindow)
		.def("main_loop", &PhysIKA::GLApp::mainLoop);

	pybind11::class_<PhysIKA::Vector3f>(m, "Vector3f")
		.def(pybind11::init())
		.def("norm", &PhysIKA::Vector3f::norm);

	py::class_<PhysIKA::SceneGraph>(m, "SceneGraph")
		.def("is_initialized", &PhysIKA::SceneGraph::isInitialized, "Return a Python dictionary")
		.def_static("get_instance", &PhysIKA::SceneGraph::getInstance, "Return an instance");
}