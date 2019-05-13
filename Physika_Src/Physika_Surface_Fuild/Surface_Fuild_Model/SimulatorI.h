#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATORI_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATORI_H_
#include "SimulationBase.h"
namespace Physika{
class SimulatorI:public SimulationBase{
public:
SimulatorI();
~SimulatorI();
void init(int argc, char** argv);
void run();
void run_cuda (int frame);
void output_obj_cuda(int frame);
void post_data ();
void clear();
SimulatorI(SimulatorI const &);
SimulatorI &operator=(SimulatorI const &);
void set_initial_constants();
void generate_origin();
void generate_mesh();
void add_properties();
//void add_index_to_vertex(int ring);
//void match_bottom_height();
void calculate_tensor();
void set_initial_conditions();
void output_obj();
//void update_midvels();
void advect_filed_values();
//void extrapolate_depth();
//void force_boundary_depth();
//void calculate_pressure();
//void update_velocity();
//void force_boundary_velocity();
//void velocity_fast_march();
void update_depth();
void release_index();
void release_properties();
float avg_edge_len;
int m_stepnum;
int m_total_steps;
MyMesh::Point m_rotate_center;
int m_output_step;
bool m_have_tensor;
float m_max_p_bs;
void prepareData ();
	float3* c_vel;
	float3* c_mvel;
	float3* c_point;
	float3* c_value0;
	float* c_bottom;
	float* c_depth;
	float* c_height;
	// bool* is_boundary_e; //?
	int* c_boundary;

	// pull#1:增加
	bool *c_once_have_water;
	bool *c_on_water_boundary;
	float *c_extrapolate_depth;
	float4 *c_tensor;
	float3 *c_wind_velocity;
	int *c_depth_boundary_type;
	float *c_depth_boundary_value;

	MyVertex* c_vertex;
	float3 (*c_vertex_rot)[3];
	int (*c_vertex_oneRing)[MAX_VERTEX];
	int (*c_vertex_nearVert)[MAX_NEAR_V];
	int3 (*c_vertex_nerbFace)[MAX_FACES];
	float3 (*c_vertex_planeMap)[MAX_FACES * 3];
	VertexOppositeHalfedge (*c_vertex_opph)[MAX_VERTEX];
}
}
#endif;
