#include "main.h"
#include <pybind11/pybind11.h>


namespace py = pybind11;

PYBIND11_MODULE(FluidEvolution, m) {
	m.def("fluidEvaluation", &fluidEvaluation);
}
