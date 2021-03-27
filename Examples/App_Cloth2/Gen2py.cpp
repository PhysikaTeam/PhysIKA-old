#include "main.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;


PYBIND11_MODULE(App_Cloth2, m) {
	m.def("AppCloth", &AppCloth);
}
