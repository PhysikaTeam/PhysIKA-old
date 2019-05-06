#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_MODELS_SIMULATOR_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_MODELS_SIMULATOR_H_
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/mymesh.h"
namespace Physika{
class Simulator {
public:
	Simulator();
	~Simulator();
	//void init(int argc, char** argv);
	//void run();
	//void run_cuda (int frame);
	//void output_obj_cuda(int frame);
	void post_data ();
	void clear();
	Simulator(Simulator const &);
	Simulator &operator=(Simulator const &);
	//void set_initial_constants();
	//void generate_origin();
	//void generate_mesh();
	void add_properties();
	void add_index_to_vertex(int ring);
	void match_bottom_height();
	//void calculate_tensor();
	//void set_initial_conditions();
	//void output_obj();
	void update_midvels();
	void advect_filed_values();
	void extrapolate_depth();
	void force_boundary_depth();
	void calculate_pressure();
	void update_velocity();
	void force_boundary_velocity();
	void velocity_fast_march();
	void update_depth();
	void release_index();
	void release_properties();

	class BoundaryCondition {
	public:
		enum DepthType { DEP_NOACTION = 0, DEP_FIXED };
		enum VelocityType { VEL_NOACTION = 0, VEL_BOUND, VEL_FIXED };
		BoundaryCondition();
		void set_depth(DepthType type, Simulator *sim, MyMesh::VertexHandle vh);
		void set_velocity(VelocityType type, Simulator *sim, MyMesh::VertexHandle vh);
		void apply_depth(float &depth);
		void apply_velocity(MyMesh::Point &velocity);
	private:
		BoundaryCondition(BoundaryCondition const &);
		BoundaryCondition &operator=(BoundaryCondition const &);
	public:
		DepthType dtype;
		VelocityType vtype;
		float dvalue0;
		MyMesh::Point vvalue0;
	};

public://private:
	//int m_situation;
	MyMesh m_mesh;
	float avg_edge_len;
	float m_depth_threshold;
	int m_stepnum;
	int m_total_steps;
	float m_dt;
	MyMesh::Point m_g;

	MyMesh m_origin;
	int m_output_step;
	bool m_have_tensor;
	float m_fric_coef;
	float m_gamma;
	float m_water_boundary_theta;
	float m_water_boundary_tension_multiplier;
	float m_max_p_bs;
	float m_wind_coef;
	bool m_enable_edit_mesh;
	bool m_output_bottom_vt;

	OpenMesh::VPropHandleT<float> m_bottom;
	OpenMesh::VPropHandleT<float> m_depth;
	OpenMesh::VPropHandleT<float> m_height; // height = base + depth
	OpenMesh::VPropHandleT<float> m_float_temp;
	OpenMesh::VPropHandleT<MyMesh::Point> m_velocity;
	OpenMesh::VPropHandleT<MyMesh::Point> m_velocity_new;
	OpenMesh::VPropHandleT<MyMesh::Point> m_midvel;
	OpenMesh::VPropHandleT<MyMesh::Point> m_vector_temp;
	OpenMesh::VPropHandleT<BoundaryCondition *> m_boundary;
	OpenMesh::VPropHandleT<int> m_label;
	OpenMesh::VPropHandleT<float> m_pressure;

	// pull#1:增加
	// 不需要的属性
	OpenMesh::VPropHandleT<MyMesh::Point> m_normal;
	// 不需传递的属性
	OpenMesh::VPropHandleT<bool> m_on_water_boundary; // 在advect末尾更新这个属性
	// 不需传递的属性
	OpenMesh::VPropHandleT<bool> m_once_have_water;
	// 新增的属性
	OpenMesh::VPropHandleT<float> m_extrapolate_depth; // 在advect末尾更新这个属性。仅在water_boundary计算插值，其它点直接复制depth
	// 不需传递的属性
	OpenMesh::VPropHandleT<float> m_pressure_gravity;
	// 不需传递的属性
	OpenMesh::VPropHandleT<float> m_pressure_surface;
	// 不需传递的属性
	OpenMesh::VPropHandleT<MyMesh::FaceHandle> m_origin_face;
	// 新增的属性
	OpenMesh::VPropHandleT<Tensor22> m_tensor;

	// Data for CUDA
	/*void prepareData ();
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
	int *c_depth_boundary_type;
	float *c_depth_boundary_value;

	MyVertex* c_vertex;
	float3 (*c_vertex_rot)[3];
	int (*c_vertex_oneRing)[MAX_VERTEX];
	int (*c_vertex_nearVert)[MAX_NEAR_V];
	int3 (*c_vertex_nerbFace)[MAX_FACES];
	float3 (*c_vertex_planeMap)[MAX_FACES * 3];
	VertexOppositeHalfedge (*c_vertex_opph)[MAX_VERTEX];*/
};
}
#endif;
