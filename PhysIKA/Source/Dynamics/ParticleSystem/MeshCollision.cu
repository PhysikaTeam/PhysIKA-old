﻿#include "MeshCollision.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "Framework/Framework/CollidableObject.h"
#include "Framework/Collision/CollidablePoints.h"
#include "Framework/Topology/NeighborQuery.h"
#include "Framework/Topology/Primitive3D.h"
#include "Framework/Topology/PrimitiveSweep3D.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(MeshCollision, TDataType)

	template<typename TDataType>
	MeshCollision<TDataType>::MeshCollision()
		: CollisionModel()
	{
	}

	template<typename TDataType>
	MeshCollision<TDataType>::~MeshCollision()
	{
		m_collidableObjects.clear();
	}

	template<typename TDataType>
	bool MeshCollision<TDataType>::isSupport(std::shared_ptr<CollidableObject> obj)
	{
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			return true;
		}
		return false;
	}


	template<typename TDataType>
	void MeshCollision<TDataType>::addCollidableObject(std::shared_ptr<CollidableObject> obj)
	{
		auto derived = std::dynamic_pointer_cast<CollidablePoints<TDataType>>(obj);
		if (obj->getType() == CollidableObject::POINTSET_TYPE)
		{
			m_collidableObjects.push_back(derived);
		}
	}



	template <typename Coord>
	__global__ void VC_Sort_Neighbors_Collide
	(
		DeviceArray<Coord> position,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> positionTri,
		NeighborList<int> neighborsTri
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= position.size()) return;
		int nbSizeTri = neighborsTri.getNeighborSize(pId);

		//if (nbSizeTri > 100)printf("NBSIZE %d\n", nbSizeTri);

		Coord pos_i = position[pId];

		//printf("nbSize: %d\n",nbSizeTri);

		for (int ne = nbSizeTri / 2 - 1; ne >= 0; ne--)
		{
			int start = ne;
			int end = nbSizeTri - 1;
			int c = start;           
			int l = 2 * c + 1;        
			int tmp = neighborsTri.getElement(pId, c);        
			for (; l <= end; c = l, l = 2 * l + 1)
			{
		
				if (l < end)
				{
					bool judge = false;
					{
						int idx1, idx2;
						idx1 = neighborsTri.getElement(pId, l);
						idx2 = neighborsTri.getElement(pId, l + 1);
						if (idx1 < 0 && idx2 < 0)
						{
							idx1 *= -1;
							idx1--;
							idx2 *= -1;
							idx2--;
							Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
							Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

							Coord normal1 = t3d1.normal().normalize();
							Coord normal2 = t3d2.normal().normalize();

							Point3D p3d(pos_i);

							Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
							Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

							Real dis1 = p3d.distance(PL1);
							Real dis2 = p3d.distance(PL2);

							if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
								judge = normal1[2] < normal2[2] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
								judge = normal1[1] < normal2[1] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON)
								judge = normal1[0] < normal2[0] ? true : false;
							else
								judge = dis1 <= dis2 ? true : false;

						}
						else judge = idx1 < idx2 ? true : false;
					}
					if (judge)
						l++;
				}
				bool judge = false;
				{
					int idx1, idx2;
					idx1 = neighborsTri.getElement(pId, l);
					idx2 = tmp;
					if (idx1 < 0 && tmp < 0)
					{
						idx1 *= -1;
						idx1--;
						idx2 *= -1;
						idx2--;
						Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
						Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

						Coord normal1 = t3d1.normal().normalize();
						Coord normal2 = t3d2.normal().normalize();

						Point3D p3d(pos_i);

						Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
						Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

						Real dis1 = p3d.distance(PL1);
						Real dis2 = p3d.distance(PL2);

						if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
							judge = normal1[2] <= normal2[2] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
							judge = normal1[1] <= normal2[1] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON)
							judge = normal1[0] <= normal2[0] ? true : false;
						else
							judge = dis1 <= dis2 ? true : false;

					}
					else judge = idx1 <= tmp ? true : false;
				}
				if (judge)
					break;
				else
				{
					neighborsTri.setElement(pId, c, neighborsTri.getElement(pId, l));
					neighborsTri.setElement(pId, l, tmp);
				}
			}
		}
		for (int ne = nbSizeTri - 1; ne > 0; ne--)
		{
			int swap_tmp = neighborsTri.getElement(pId, 0);
			neighborsTri.setElement(pId, 0, neighborsTri.getElement(pId, ne));
			neighborsTri.setElement(pId, ne, swap_tmp);
			int start = 0;
			int end = ne - 1;
			int c = start;
			int l = 2 * c + 1;
			int tmp = neighborsTri.getElement(pId, c);
			for (; l <= end; c = l, l = 2 * l + 1)
			{
			
				if (l < end)
				{
					bool judge = false;
					{
						int idx1, idx2;
						idx1 = neighborsTri.getElement(pId, l);
						idx2 = neighborsTri.getElement(pId, l + 1);
						if (neighborsTri.getElement(pId, l) < 0 && neighborsTri.getElement(pId, l + 1) < 0)
						{
							idx1 *= -1;
							idx1--;
							idx2 *= -1;
							idx2--;
							Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
							Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

							Coord normal1 = t3d1.normal().normalize();
							Coord normal2 = t3d2.normal().normalize();

							Point3D p3d(pos_i);

							Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
							Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

							Real dis1 = p3d.distance(PL1);
							Real dis2 = p3d.distance(PL2);

							if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
								judge = normal1[2] < normal2[2] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
								judge = normal1[1] < normal2[1] ? true : false;
							else if (abs(dis1 - dis2) < EPSILON)
								judge = normal1[0] < normal2[0] ? true : false;
							else
								judge = dis1 < dis2 ? true : false;

						}
						else judge = neighborsTri.getElement(pId, l) < neighborsTri.getElement(pId, l + 1) ? true : false;
					}
					if (judge)
						l++;
				}
				bool judge = false;
				{
					int idx1, idx2;
					idx1 = neighborsTri.getElement(pId, l);
					idx2 = tmp;
					if (neighborsTri.getElement(pId, l) < 0 && tmp < 0)
					{
						idx1 *= -1;
						idx1--;
						idx2 *= -1;
						idx2--;
						Triangle3D t3d1(positionTri[m_triangle_index[idx1][0]], positionTri[m_triangle_index[idx1][1]], positionTri[m_triangle_index[idx1][2]]);
						Triangle3D t3d2(positionTri[m_triangle_index[idx2][0]], positionTri[m_triangle_index[idx2][1]], positionTri[m_triangle_index[idx2][2]]);

						Plane3D PL1(positionTri[m_triangle_index[idx1][0]], t3d1.normal());
						Plane3D PL2(positionTri[m_triangle_index[idx2][0]], t3d2.normal());

						Coord normal1 = t3d1.normal().normalize();
						Coord normal2 = t3d2.normal().normalize();

						Point3D p3d(pos_i);

						Real dis1 = p3d.distance(PL1);
						Real dis2 = p3d.distance(PL2);

						if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON && abs(normal1[1] - normal2[1]) < EPSILON)
							judge = normal1[2] <= normal2[2] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON && abs(normal1[0] - normal2[0]) < EPSILON)
							judge = normal1[1] <= normal2[1] ? true : false;
						else if (abs(dis1 - dis2) < EPSILON)
							judge = normal1[0] <= normal2[0] ? true : false;
						else
							judge = dis1 <= dis2 ? true : false;

					}
					else judge = neighborsTri.getElement(pId, l) <= tmp ? true : false;
				}
				if (judge)
					break;
				else
				{
					neighborsTri.setElement(pId, c, neighborsTri.getElement(pId, l));
					neighborsTri.setElement(pId, l, tmp);
				}
			}
		}
		return;
	}





	template<typename Real, typename Coord>
	__global__ void K_CD_mesh2(
		DeviceArray<Coord> points,
		DeviceArray<Coord> pointsTri,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> vels,
		DeviceArray<int> flip,
		NeighborList<int> neighborsTriangle,
		Real radius,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;
		int nbSizeTri = neighborsTriangle.getNeighborSize(pId);

		Coord pos_i = points[pId];
		Real nearest_distance = 1.0;
		int nearest_triangle = 0;
		int nj;

		Coord pos_i_next = points[pId] + vels[pId] * dt;
		Coord vel_tmp = vels[pId];

		Coord new_pos(0);
		Real weight(0);


		Coord old_pos = pos_i;
		for (int it = 0; it < 1; it++)
		{
			Coord new_pos(0);
			Real weight(0);
			for (int ne = 0; ne < nbSizeTri; ne++)
			{
				int j = neighborsTriangle.getElement(pId, ne);
				if (j >= 0) continue;
				j *= -1;
				j--;
				Triangle3D t3d(pointsTri[m_triangle_index[j][0]], pointsTri[m_triangle_index[j][1]], pointsTri[m_triangle_index[j][2]]);
				
				Point3D p3d(pos_i);
				Point3D nearest_point = p3d.project(t3d);

				Real r = (p3d.distance(t3d));

				Coord n = t3d.normal();
				if (n.norm() > EPSILON)
				{
					n.normalize();
				}



				if ((abs(r) < radius) && abs(r) > EPSILON)  
				{
					Point3D pt_neartest = nearest_point;
					Coord3D pt_norm = -pt_neartest.origin + p3d.origin;
					pt_norm /= abs(r);
					new_pos += pt_neartest.origin + radius * pt_norm;
					weight += 1.0;
				}

			}
			if (weight < EPSILON)
			{
			}
			else
			{
				pos_i = new_pos / weight;
			}
		}
		points[pId] = pos_i;
		vels[pId] += (pos_i - old_pos) / dt;
	}



	template<typename Real, typename Coord>
	__global__ void K_ComputeTarget(
		DeviceArray<Coord> oldPoints,
		DeviceArray<Coord> newPoints, 
		DeviceArray<Real> weights)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= oldPoints.size()) return;

		if (weights[pId] > EPSILON)
		{
			newPoints[pId] /= weights[pId];
		}
		else
			newPoints[pId] = oldPoints[pId];

	}

	template<typename Real, typename Coord>
	__global__ void K_ComputeVelocity(
		DeviceArray<Coord> initPoints,
		DeviceArray<Coord> curPoints,
		DeviceArray<Coord> velocites,
		Real dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= velocites.size()) return;

		Coord add_vel = (curPoints[pId] - initPoints[pId] + velocites[pId] * dt) / dt;
		velocites[pId] += add_vel;
	}

	//Continuous collision detection between points and triangles
	template<typename Real, typename Coord>
	__global__ void K_CCD_MESH(
		DeviceArray<Coord> particle_position,
		DeviceArray<Coord> particle_velocity,
		DeviceArray<Coord> particle_position_previous,
		DeviceArray<Coord> triangle_vertex,
		DeviceArray<Coord> triangle_vertex_previous,
		DeviceArray<TopologyModule::Triangle> triangle_index,
		NeighborList<int> triangle_neighbors,
		Real threshold,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= particle_position.size()) return;

		
		int nbrSize = triangle_neighbors.getNeighborSize(pId);

		Coord pos_i = particle_position[pId];
		Coord pos_i_old = particle_position_previous[pId];
 		Coord vel_i = particle_velocity[pId];

		Point3D start_point(particle_position[pId]);
		Point3D end_point(particle_position[pId]);
		PointSweep3D ptSweep(start_point, end_point);

		Real t = 100.0f;
		Real min_t = 10.0f;
		bool bIntersected = false;
		int min_j = -1;

		int interNum = 0;
		Coord3D total_pos(0);
		Coord3D delta_pos(0);

		
		
		for (int ne = 0; ne < nbrSize; ne++)
		{
			int j = triangle_neighbors.getElement(pId, ne);
			if (j >= 0) continue;
			j *= -1;
			j--;
			
			Point3D p3d(pos_i);
			Point3D p3dp(pos_i);

			Triangle3D t3d(triangle_vertex[triangle_index[j][0]], triangle_vertex[triangle_index[j][1]], triangle_vertex[triangle_index[j][2]]);
			Real min_distance = abs(p3d.distance(t3d));
			if (ne < nbrSize - 1 && triangle_neighbors.getElement(pId, ne + 1) < 0)
			{
				int jn;
				do
				{
					jn = triangle_neighbors.getElement(pId, ne + 1);
					if (jn > 0) break;
					jn *= -1; jn--;

					Triangle3D t3d_n(triangle_vertex[triangle_index[jn][0]], triangle_vertex[triangle_index[jn][1]], triangle_vertex[triangle_index[jn][2]]);
					if ((t3d.normal().cross(t3d_n.normal())).norm() > EPSILON * t3d_n.normal().norm() * t3d.normal().norm()) break;

					if (abs(p3d.distance(t3d_n)) < abs(min_distance))
					{
						j = jn;
						min_distance = abs(p3d.distance(t3d_n));
					}
				
					ne++;
				} while (ne < nbrSize - 1);
			}

			Triangle3D start_triangle(triangle_vertex_previous[triangle_index[j][0]], triangle_vertex_previous[triangle_index[j][1]], triangle_vertex_previous[triangle_index[j][2]]);
			Triangle3D end_triangle(triangle_vertex[triangle_index[j][0]], triangle_vertex[triangle_index[j][1]], triangle_vertex[triangle_index[j][2]]);

			TriangleSweep3D triSweep(start_triangle, end_triangle);
			
			typename Triangle3D::Param baryc;
			int num = ptSweep.intersect(triSweep, baryc, t, threshold);

			if (num > 0)
			{
				interNum++;
				bool tmp = false;
				Coord3D proj_j = end_triangle.computeLocation(baryc);

				Coord3D dir_j = start_point.origin - proj_j;
				if (dir_j.norm() > REAL_EPSILON)
				{
					dir_j.normalize();
				}
				if (false)//(p3d.distance(end_triangle) < 0)
				{
					dir_j *= -1;
					tmp = true;
				}

				Coord newpos = proj_j + dir_j * threshold;
				




				total_pos += newpos;// end_triangle.computeLocation(tParam) + threshold * end_triangle.normal();
				auto d_pos_i = newpos - pos_i;
				if (d_pos_i.norm() > threshold )
				{
					d_pos_i *= 0.0;// threshold / d_pos_i.norm();
				}
				delta_pos += d_pos_i;
			}


		}

		if (interNum > 0)
		{
			
			total_pos /= interNum;
			delta_pos /= interNum;
			
			particle_position[pId] = total_pos;

			Real norm_v = vel_i.norm();
			Coord old_vel = vel_i;

			vel_i += delta_pos / dt;
			
			//if(vel_i.norm() > norm_v)
			//  vel_i = vel_i / vel_i.norm() * norm_v;
			

			particle_velocity[pId] = vel_i;
		}
		return;
	}

	template<typename TDataType>
	void MeshCollision<TDataType>::doCollision()
	{
		Real radius = 0.005;


		if (m_position.getElementCount() == 0) return;
		
		printf("to resize %d %d\n", m_position_previous.size(), m_position.getElementCount());

		
		
		

		if (m_position_previous.isEmpty() || m_position_previous.size() != m_position.getElementCount())
		{
		//	m_flip.setElementCount(m_position.getElementCount());
			
			posBuf.resize(m_position.getElementCount());
			weights.resize(m_position.getElementCount());
			init_pos.resize(m_position.getElementCount());

			m_position_previous.resize(m_position.getElementCount());
			Function1Pt::copy(m_position_previous, m_position.getValue());
			printf("resize finished\n");
		}
		
		//auto m_points = m_position.getValue();
		//auto m_points_tri = m_triangle_vertex.getValue();
		//auto m_tri_idx = m_triangle_index.getValue();
		//auto m_vels = m_velocity.getValue();
		//auto m_neighbor_tri = m_neighborhood_tri.getValue();
	//	auto flip = m_flip.getValue();

		Function1Pt::copy(init_pos, m_position.getValue());

		//uint pDims = cudaGridSize(m_position.getValue().size(), BLOCK_SIZE);

	//	flip.reset();
		
		//cuExecute
		int total_num = m_position.getValue().size();
		
		//VC_Sort_Neighbors_Collide <Real, Coord> << <pDims, BLOCK_SIZE >> > (
		/**/

		cuSynchronize();

		cuExecute(total_num, VC_Sort_Neighbors_Collide,
			m_position.getValue(),
			m_triangle_index.getValue(),
			m_triangle_vertex.getValue(),
			m_neighborhood_tri.getValue()
			);
		
		//K_CCD_MESH <Real, Coord> << <pDims, BLOCK_SIZE >> > (
		
		cuExecute(total_num, K_CCD_MESH,
			m_position.getValue(),
			m_velocity.getValue(),
			m_position_previous,
			m_triangle_vertex.getValue(),
			m_triangle_vertex_previous,
			m_triangle_index.getValue(),
			m_neighborhood_tri.getValue(),
			radius,
			getParent()->getDt()
			);
		
		

// 		K_CD_mesh2 << <pDims, BLOCK_SIZE >> > (
// 			m_position.getValue(),
// 			m_triangle_vertex.getValue(),
// 			m_triangle_index.getValue(),
// 			m_velocity.getValue(),
// 			m_flip.getValue(),
// 			m_neighborhood_tri.getValue(),
// 			radius,
// 			getParent()->getDt()
// 			);

		Function1Pt::copy(m_triangle_vertex_previous, m_triangle_vertex.getValue());
		Function1Pt::copy(m_position_previous, m_position.getValue());

	}

	
	template<typename TDataType>
	bool MeshCollision<TDataType>::initializeImpl()
	{
		m_triangle_vertex_previous.resize(m_triangle_vertex.getElementCount());
		Function1Pt::copy(m_triangle_vertex_previous, m_triangle_vertex.getValue());

		if (m_position.getElementCount() == 0)
			return true;
		if (m_flip.isEmpty())
		{
			m_flip.setElementCount(m_position.getElementCount());
		}
		posBuf.resize(m_position.getElementCount());
		weights.resize(m_position.getElementCount());
		init_pos.resize(m_position.getElementCount());

		m_position_previous.resize(m_position.getElementCount());
		Function1Pt::copy(m_position_previous, m_position.getValue());

		

		return true;
	}

}