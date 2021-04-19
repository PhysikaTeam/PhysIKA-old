#include "SatDataCloud_Lite.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <string>

namespace py = pybind11;


PYBIND11_MODULE(SatImageTyphoon, m) {
    py::class_<SatDataCloud>(m, "SatDataCloud")
        .def(py::init<>())
        .def("Run", static_cast<void (SatDataCloud::*)(const vector<string>& , const string& , const string& , int, int)>(&SatDataCloud::Run));
}
