#include "SimulatorBase.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexonvertex.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexofmesh.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/cvfem.h"
#include "Physika_Surface_Fuild/Surface_Smooth/smooth.h"
#include "Physika_Surface_Fuild/Surface_Utilities/boundrecorder.h"
#include "Physika_Surface_Fuild/Surface_Utilities/windowstimer.h"
#include <vector>
#include <queue>
#include <deque>
namespace Physika{
SimulatonBase::BoundaryCondition::BoundaryCondition()
:dtype(DEP_NOACTION), vtype(VEL_NOACTION) { }

void SimulatonBase::BoundaryCondition::set_depth(DepthType type, Simulator *sim, MyMesh::VertexHandle vh) {
	this->dtype = type;
	switch (type) {
	case DEP_FIXED:
		dvalue0 = sim->m_mesh.property(sim->m_depth, vh);
		break;
	case DEP_NOACTION:
		break;
	default:
		break;
	};
}

void SimulatonBase::BoundaryCondition::set_velocity(VelocityType type, Simulator *sim, MyMesh::VertexHandle vh) {
	this->vtype = type;
	switch (type) {
	case VEL_BOUND:
	{
					  IndexOnVertex *index = sim->m_mesh.data(vh).index;
					  MyMesh::Point in_direct(0, 0, 0);
					  MyMesh::Point out_direct(0, 0, 0);
					  for (auto vih_it = sim->m_mesh.cvih_iter(vh); vih_it.is_valid(); ++vih_it) {
						  if (sim->m_mesh.is_boundary(*vih_it)) {
							  MyMesh::Point b(sim->m_mesh.point(sim->m_mesh.from_vertex_handle(*vih_it)));
							  in_direct += -index->plane_map(b);
						  }
					  }
					  for (auto voh_it = sim->m_mesh.cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
						  if (sim->m_mesh.is_boundary(*voh_it)) {
							  MyMesh::Point c(sim->m_mesh.point(sim->m_mesh.to_vertex_handle(*voh_it)));
							  out_direct += index->plane_map(c);
						  }
					  }
					  vvalue0 = (in_direct + out_direct).normalized();
					  // Todo: 处理介于直角与平边之间的情况
					  if ((in_direct.normalized() | out_direct.normalized()) < 0.5f)
						  vvalue0 = MyMesh::Point(0, 0, 0);
	}
		break;
	case VEL_FIXED:
		vvalue0 = sim->m_mesh.property(sim->m_velocity, vh);
		break;
	case VEL_NOACTION:
		break;
	default:
		break;
	};
}

void SimulatonBase::BoundaryCondition::apply_depth(float &depth) {
	switch (dtype) {
	case DEP_FIXED:
		depth = dvalue0;
		break;
	case DEP_NOACTION:
		break;
	default:
		break;
	};
}

void Simulator::BoundaryCondition::apply_velocity(MyMesh::Point &velocity) {
	switch (vtype) {
	case VEL_BOUND:
		velocity = (velocity | vvalue0) * vvalue0;
		break;
	case VEL_FIXED:
		velocity = vvalue0;
		break;
	case VEL_NOACTION:
		break;
	default:
		break;
	};
}
}
