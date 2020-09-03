#include "MeshBoundary.h"
#include "Core/Utility.h"
#include "Framework/Framework/Log.h"
#include "Framework/Framework/Node.h"
#include "Dynamics/ParticleSystem/BoundaryConstraint.h"

#include "Framework/Topology/DistanceField3D.h"
#include "Framework/Topology/TriangleSet.h"
#include "Framework/Topology/NeighborQuery.h"

namespace PhysIKA
{
	IMPLEMENT_CLASS_1(MeshBoundary, TDataType)



	template<typename Real, typename Coord>
	__global__ void K_CD_mesh(
		DeviceArray<Coord> points,
		DeviceArray<Coord> pointsTri,
		DeviceArray<TopologyModule::Triangle> m_triangle_index,
		DeviceArray<Coord> vels,
		NeighborList<int> neighborsTriangle,
		Real radius,
		Real dt
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;
		int nbSizeTri = neighborsTriangle.getNeighborSize(pId);
	//	if (pId == 0)
	//		printf("******************************************%d\n",points.size());

		Coord pos_i = points[pId];
		Real nearest_distance = 1.0;
		int nj;
		if (vels[pId].norm() > radius / dt)
			vels[pId] = vels[pId] / vels[pId].norm() * radius / dt;
		Coord vel_tmp = vels[pId];

		
		Coord old_pos = pos_i;
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
				if (((r) < radius) && abs(r) > EPSILON)
				{
					Point3D pt_neartest = nearest_point;
					Coord3D pt_norm = -pt_neartest.origin + p3d.origin;
					pt_norm /= (r);
					new_pos += pt_neartest.origin + radius * pt_norm;
					weight += 1.0;
				}

		}
		if (weight > EPSILON)
		{
			pos_i = new_pos / weight;
			Coord dir = (pos_i - old_pos) / (pos_i - old_pos).norm();
			vels[pId] -= vels[pId].dot(dir) * dir;

			//printf("%.3lf %.3lf %.3lf *** %.3lf %.3lf %.3lf \n", pos_i[0], pos_i[1], pos_i[2], old_pos[0], old_pos[1], old_pos[2]);
		}
		//points[pId] = Coord(0);
		points[pId] = pos_i;

		
		//printf("%.3lf %.3lf %.3lf *** %.3lf %.3lf %.3lf \n", points[pId][0], points[pId][1], points[pId][2], vels[pId][0], vels[pId][1], vels[pId][2]);
	}

	template<typename Coord>
	__global__ void TEST_mesh(
		DeviceArray<Coord> points,
		DeviceArray<Coord> vels
	)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= points.size()) return;
		//printf("YES\n");
		if (points[pId].norm() > EPSILON|| vels[pId].norm()) printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@ERROR\n");
	}

	template<typename TDataType>
	MeshBoundary<TDataType>::MeshBoundary()
		: Node()
	{
	}

	template<typename TDataType>
	MeshBoundary<TDataType>::~MeshBoundary()
	{
	}

	template<typename TDataType>
	void MeshBoundary<TDataType>::loadMesh(std::string filename)
	{
		printf("inside load\n");
		auto boundary = std::make_shared<TriangleSet<TDataType>>();
		boundary->loadObjFile(filename);
		m_obstacles.push_back(boundary);
		printf("outside load\n");
	}

	template<typename TDataType>
	bool MeshBoundary<TDataType>::initialize()
	{
		return true;
	}

	template<typename TDataType>
	bool MeshBoundary<TDataType>::resetStatus()
	{
	//	printf("RESET1\n");
		int sum_tri_index = 0;
		int sum_tri_pos = 0;
		int sum_poi_pos = 0;

		for (size_t t = 0; t < m_obstacles.size(); t++)
		{
			sum_tri_index += m_obstacles[t]->getTriangles()->size();
			sum_tri_pos += m_obstacles[t]->getPoints().size();
		}
	//	printf("RESET2\n");
		triangle_index.setElementCount(sum_tri_index);
		triangle_positions.setElementCount(sum_tri_pos);

		int start_pos = 0;
		int start_tri = 0;
	//	printf("RESET3\n");
		for (size_t t = 0; t < m_obstacles.size(); t++)
		{
			DeviceArray<Coord> posTri = m_obstacles[t]->getPoints();
			DeviceArray<Triangle>* idxTri = m_obstacles[t]->getTriangles();
			int num_p = posTri.size();
			int num_i = idxTri->size();
			cudaMemcpy(triangle_positions.getValue().getDataPtr() + start_pos, posTri.getDataPtr(), num_p * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(triangle_index.getValue().getDataPtr() + start_tri, idxTri->getDataPtr(), num_i * sizeof(Triangle), cudaMemcpyDeviceToDevice);
			start_pos += num_p;
			start_tri += num_i;
		}

		//printf("RESET4\n");
		auto pSys = this->getParticleSystems();
		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			sum_poi_pos += posFd->getElementCount();
		}
		point_positions.setElementCount(sum_poi_pos);
		point_velocities.setElementCount(sum_poi_pos);
		//printf("RESET5\n");
		int start_point = 0;
		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();
			int num = posFd->getElementCount();
			cudaMemcpy(point_positions.getValue().getDataPtr() + start_point, posFd->getValue().getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(point_velocities.getValue().getDataPtr() + start_point, velFd->getValue().getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start_point += num;
		}
	//	printf("RESET6\n");
		radius.setValue(0.005);
		m_nbrQuery = this->template addComputeModule<NeighborQuery<TDataType>>("neighborhood");
		radius.connect(m_nbrQuery->inRadius());
		point_positions.connect(m_nbrQuery->inPosition());
		triangle_positions.connect(m_nbrQuery->inTrianglePosition());
		printf("tri pos size: %d\n", triangle_positions.getElementCount());
		printf("tri idx size: %d\n", triangle_index.getElementCount());

		triangle_index.connect(m_nbrQuery->inTriangleIndex());
		m_nbrQuery->initialize();
		m_nbrQuery->compute();

		//printf("RESET7\n");
		return Node::resetStatus();
	}

	template<typename TDataType>
	void MeshBoundary<TDataType>::advance(Real dt)
	{
		
		int sum_poi_pos = 0;
		auto pSys = this->getParticleSystems();
		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			sum_poi_pos += posFd->getElementCount();
		}
		point_positions.setElementCount(sum_poi_pos);
		point_velocities.setElementCount(sum_poi_pos);

		int start_point = 0;
		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();
			int num = posFd->getElementCount();
			cudaMemcpy(point_positions.getValue().getDataPtr() + start_point, posFd->getValue().getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(point_velocities.getValue().getDataPtr() + start_point, velFd->getValue().getDataPtr(), num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start_point += num;
		}
	//	printf("compute nbr\n");
		m_nbrQuery->compute();
		cuSynchronize();
		cuSynchronize();

		DeviceArray<Coord>& poss = point_positions.getValue();
		DeviceArray<Coord>& vels = point_velocities.getValue();

		uint pDims = cudaGridSize(point_positions.getValue().size(), BLOCK_SIZE);
		K_CD_mesh << <pDims, BLOCK_SIZE >> > (
						poss,
						triangle_positions.getValue(),
						triangle_index.getValue(),
						vels,
						m_nbrQuery->outNeighborhood()->getValue(),
			 			radius.getValue(),
			 			dt
			 			);

		cuSynchronize();

		//TEST_mesh << <pDims, BLOCK_SIZE >> > (
		//	poss,
		//	vels
		//	);

		cuSynchronize();

		start_point = 0;
		//point_positions.getValue().reset();
		//poss.reset();

		for (int i = 0; i < pSys.size(); i++)
		{
			DeviceArrayField<Coord>* posFd = pSys[i]->currentPosition();
			DeviceArrayField<Coord>* velFd = pSys[i]->currentVelocity();
			
			int num = posFd->getElementCount();
			//printf("%d\n", num);

			cudaMemcpy(posFd->getValue().getDataPtr(), poss.getDataPtr() + start_point, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			cudaMemcpy(velFd->getValue().getDataPtr(), vels.getDataPtr() + start_point, num * sizeof(Coord), cudaMemcpyDeviceToDevice);
			start_point += num;
		}

	}

	
}
