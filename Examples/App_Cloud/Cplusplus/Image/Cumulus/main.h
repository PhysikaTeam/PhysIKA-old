#pragma once
#include <string>
#include <vector>
using namespace std;

struct EXT_INFO {
	vector<float> sun_color;
	vector<int> img_WH;
	int num_vertices;
	int num_faces;
};


void half_mesh(string& upload_filename);
void mesh_deform(string& sim_filename);
void sim_cloud(string upload_filename, string sim_filename);

vector<float> get_sun_color();
vector<int> get_img_WH();
int get_num_vertices();
int get_num_faces();