//
// Created by sjeske on 1/22/20.
//
#include "common.h"

#include <SPlisHSPlasH/TimeStep.h>
#include <SPlisHSPlasH/DFSPH_K/SimulationDataDFSPH_K.h>
#include <SPlisHSPlasH/DFSPH_K/TimeStepDFSPH_K.h>

#include <pybind11/pybind11.h>

namespace py = pybind11;

void DFSPH_KModule(py::module m_sub) {
	// ---------------------------------------
	// Class Simulation Data DFSPH_K
	// ---------------------------------------
	py::class_<SPH::SimulationDataDFSPH_K>(m_sub, "SimulationDataDFSPH_K")
		.def(py::init<>())
		.def("init", &SPH::SimulationDataDFSPH_K::init)
		.def("cleanup", &SPH::SimulationDataDFSPH_K::cleanup)
		.def("reset", &SPH::SimulationDataDFSPH_K::reset)
		.def("performNeighborhoodSearchSort", &SPH::SimulationDataDFSPH_K::performNeighborhoodSearchSort)

		.def("getFactor", (const Real(SPH::SimulationDataDFSPH_K::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDFSPH_K::getFactor))
		// .def("getFactor", (Real& (SPH::SimulationDataDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDFSPH::getFactor)) // TODO: wont work by reference
		.def("setFactor", &SPH::SimulationDataDFSPH_K::setFactor)

		.def("getKappa", (const Real(SPH::SimulationDataDFSPH_K::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDFSPH_K::getKappa))
		// .def("getKappa", (Real& (SPH::SimulationDataDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDFSPH::getKappa)) // TODO: wont work by reference
		.def("setKappa", &SPH::SimulationDataDFSPH_K::setKappa)

		.def("getKappaV", (const Real(SPH::SimulationDataDFSPH_K::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDFSPH_K::getKappaV))
		// .def("getKappaV", (Real& (SPH::SimulationDataDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDFSPH::getKappaV)) // TODO: wont work by reference
		.def("setKappaV", &SPH::SimulationDataDFSPH_K::setKappaV)

		.def("getDensityAdv", (const Real(SPH::SimulationDataDFSPH_K::*)(const unsigned int, const unsigned int)const)(&SPH::SimulationDataDFSPH_K::getDensityAdv))
		// .def("getDensityAdv", (Real& (SPH::SimulationDataDFSPH::*)(const unsigned int, const unsigned int))(&SPH::SimulationDataDFSPH::getDensityAdv)) // TODO: wont work by reference
		.def("setDensityAdv", &SPH::SimulationDataDFSPH_K::setDensityAdv);

	// ---------------------------------------
	// Class Time Step DFSPH_K
	// ---------------------------------------
	py::class_<SPH::TimeStepDFSPH_K, SPH::TimeStep>(m_sub, "TimeStepDFSPH_K")
		.def_readwrite_static("SOLVER_ITERATIONS_V", &SPH::TimeStepDFSPH_K::SOLVER_ITERATIONS_V)
		.def_readwrite_static("MAX_ITERATIONS_V", &SPH::TimeStepDFSPH_K::MAX_ITERATIONS_V)
		.def_readwrite_static("MAX_ERROR_V", &SPH::TimeStepDFSPH_K::MAX_ERROR_V)
		.def_readwrite_static("USE_DIVERGENCE_SOLVER", &SPH::TimeStepDFSPH_K::USE_DIVERGENCE_SOLVER)

		.def(py::init<>());
}