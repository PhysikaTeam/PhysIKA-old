#include "PyRendering.h"

#include "Rendering/PointRenderModule.h"

void declare_point_render_module(py::module &m) {
	using Class = PhysIKA::PointRenderModule;
	using VisualModule = PhysIKA::VisualModule;
	std::string pyclass_name = std::string("PointRenderModule");
	py::class_<Class, VisualModule, std::shared_ptr<Class>>(m, pyclass_name.c_str(), py::buffer_protocol(), py::dynamic_attr())
		.def(py::init<>());
}

void pybind_rendering(py::module& m)
{
	declare_point_render_module(m);
}