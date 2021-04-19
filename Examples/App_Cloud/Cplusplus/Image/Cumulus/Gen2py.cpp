#include "main.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

//extern EXT_INFO ext_info;

PYBIND11_MODULE(Sim_Cloud, m) {
	m.def("sim_cloud", &sim_cloud);
	m.def("get_sun_color", &get_sun_color);
	m.def("get_img_WH", &get_img_WH);
	m.def("get_num_vertices", &get_num_vertices);
	m.def("get_num_faces", &get_num_faces);
	//m.attr("sun_color") = ext_info.sun_color;
	//m.attr("img_WH") = ext_info.img_WH;
	//m.attr("num_vertices") = ext_info.num_vertices;
	//m.attr("num_faces") = ext_info.num_faces;
}
