#ifndef SIMULATOR_H
#define SIMULATOR_H

#include "mymesh.h"

class Simulator {
public:
	Simulator();
	~Simulator();
	void getheight();
	void init(int argc, char** argv);
	void init();
	void belowwatermodel();
	void runoneframe();
	void clear();               
	Simulator(Simulator const &);
	Simulator &operator=(Simulator const &);
	void set_initial_constants(bool m_have_tensor, float m_fric_coef, float m_gamma, float m_dt, float g);
	void set_initial_constants();
	void generate_origin(int MRES_X, int MRES_Z, std::vector<float> bt, float grid_size);
	void generate_origin();
	void generate_mesh(int situation, int times);
	void generate_mesh();
	void add_properties();
	void add_index_to_vertex(int ring);
	void match_bottom_height();
	void calculate_tensor();
	void set_initial_conditions(std::vector<float> hi, std::vector<std::vector<float>> v);
	void set_initial_conditions(std::vector<float> hi, std::vector<float> v);
	void set_initial_conditions();
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
	int m_situation;
	MyMesh m_mesh;
	float avg_edge_len;
	float m_depth_threshold;
	//int m_stepnum;
	//int m_total_steps;
	float m_dt;
	MyMesh::Point m_g;

	MyMesh m_origin;
	//int m_output_step;
	bool m_have_tensor;
	float m_fric_coef;
	float m_gamma;
	float m_water_boundary_theta;
	float m_water_boundary_tension_multiplier;
	float m_max_p_bs;
	size_t x_cells;
	size_t z_cells;
    std::vector<float> bottom;
	std::vector<float> height;

	OpenMesh::VPropHandleT<float> m_bottom;
	OpenMesh::VPropHandleT<float> m_depth;
	OpenMesh::VPropHandleT<float> m_height; 
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


};

#endif
