#include <tcdsmModeler/Gen2py_helper.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

PYBIND11_MODULE(tcdsmModeler, m) {
    m.def("execute", &execute);
}
