#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATIONBASE_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_FUILD_MODEL_SIMULATIONBASE_H_
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/mymesh.h"
namespace Physika{
class SimulationBase{
public:
	SimulatorBase();
	~SimulatorBase();
	virtual void init(int argc, char** argv)=0;
	virtual void run()=0;
	virtual void run_cuda (int frame)=0;
	virtual void output_obj_cuda(int frame)=0;
	virtual void post_data ()=0;
	virtual void clear()=0;
	virtual void set_initial_constants()=0;
	virtual void generate_origin()=0;
	virtual void generate_mesh()=0;
	virtual void add_properties()=0;
	virtual void add_index_to_vertex(int ring)=0;
	virtual void match_bottom_height()=0;
	virtual void calculate_tensor()=0;
	virtual void set_initial_conditions()=0;
	virtual void output_obj()=0;
	virtual void output_wind()=0;
	virtual void edit_mesh()=0;
	virtual void edit_mesh_update_normal()=0;
	virtual void edit_mesh_update_index()=0;
	virtual void edit_mesh_update_tensor()=0;
	virtual void update_midvels()=0;
	virtual void advect_filed_values()=0;
	virtual void extrapolate_depth()=0;
	virtual void force_boundary_depth()=0;
	virtual void calculate_pressure()=0;
	virtual void update_velocity()=0;
	virtual void force_boundary_velocity()=0;
	virtual void velocity_fast_march()=0;
	virtual void update_depth()=0;
	virtual void release_index()=0;
	virtual void release_properties()=0;
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
