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
	virtual void add_index_to_vertex(int ring);
	virtual void match_bottom_height();
	virtual void calculate_tensor();
	virtual void set_initial_conditions();
	virtual void output_obj();
	virtual void output_wind();
	virtual void edit_mesh();
	virtual void edit_mesh_update_normal();
	virtual void edit_mesh_update_index();
	virtual void edit_mesh_update_tensor();
	virtual void update_midvels();
	virtual void advect_filed_values();
	virtual void extrapolate_depth();
	virtual void force_boundary_depth();
	virtual void calculate_pressure();
	virtual void update_velocity();
	virtual void force_boundary_velocity();
	virtual void velocity_fast_march();
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
}
#endif
