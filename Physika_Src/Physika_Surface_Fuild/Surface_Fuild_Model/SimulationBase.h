#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATIONBASE_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATIONBASE_H_
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/mymesh.h"
namespace Physika{
class SimulationBase{
public:
	SimulatorBase();
	~SimulatorBase();
	virtual void init(int argc, char** argv);
	virtual void run();
	virtual void run_cuda (int frame);
	virtual void output_obj_cuda(int frame);
	virtual void post_data ();
	virtual void clear();
	virtual void set_initial_constants();
	virtual void generate_origin();
	virtual void generate_mesh();
	virtual void add_properties();
	void add_index_to_vertex(int ring);
	void match_bottom_height();
	virtual void calculate_tensor();
	virtual void set_initial_conditions();
	virtual void output_obj();
	virtual void output_wind();
	virtual void edit_mesh();
	virtual void edit_mesh_update_normal();
	virtual void edit_mesh_update_index();
	virtual void edit_mesh_update_tensor();
	void update_midvels();
	void advect_filed_values();
	void extrapolate_depth();
	void force_boundary_depth();
	void calculate_pressure();
	void update_velocity();
	void force_boundary_velocity();
	void velocity_fast_march();
	virtual void update_depth();
	virtual void release_index();
	virtual void release_properties();
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
	MyMesh m_mesh;
	MyMesh m_origin;
	float m_dt;
	float m_depth_threshold;
	MyMesh::Point m_g;
	float m_gamma;
	float m_water_boundary_tension_multiplier;
	float m_water_boundary_theta;
	float m_wind_coef;
	//bool m_have_tensor;
	float m_fric_coef;
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
	OpenMesh::VPropHandleT<MyMesh::Point> m_normal;
	OpenMesh::VPropHandleT<bool> m_on_water_boundary; 
	OpenMesh::VPropHandleT<bool> m_once_have_water;
	OpenMesh::VPropHandleT<float> m_extrapolate_depth; // 在advect末尾更新这个属性。仅在water_boundary计算插值，其它点直接复制depth
	OpenMesh::VPropHandleT<float> m_pressure_gravity;
	OpenMesh::VPropHandleT<float> m_pressure_surface;
	OpenMesh::VPropHandleT<MyMesh::FaceHandle> m_origin_face;
	OpenMesh::VPropHandleT<Tensor22> m_tensor;
	OpenMesh::VPropHandleT<MyMesh::Point> m_wind_velocity;
}
#endif
