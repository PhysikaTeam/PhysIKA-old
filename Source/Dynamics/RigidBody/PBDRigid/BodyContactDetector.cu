#include "Dynamics/RigidBody/PBDRigid/BodyContactDetector.h"

#include "Core/Utility/Function1Pt.h"

namespace PhysIKA
{



	__global__ void HFBDetect_detectContactPairPerRigidPoint(
		ContactInfo<double>* contacts,
		//DeviceDArray<int> beInContact,
		int *counter,
		DeviceHeightField1d grid,
		DeviceArray<Vector3f> vertices,
		RigidInteractionInfo rigidinfo,
		int heightfieldid, int rigidid, float mu,
		float boundaryThreashold)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= vertices.size()) return;



		auto vertex = rigidinfo.rotation.rotate(vertices[tid]) + rigidinfo.position;

		float h = grid.get(vertex[0], vertex[2]);
		float depth = h - vertex[1];

		if (depth >= -boundaryThreashold)
		{

			int id = atomicAdd(counter, 1);

			//printf(" ID:  %d,  %d\n", tid, id);

			Vector3d pointnormal = grid.heightFieldNormal(vertex[0], vertex[2]);
			(contacts + id)->point0 = Vector3d(vertex[0], h, vertex[2]);
			(contacts + id)->point1 = Vector3d(vertex[0], vertex[1], vertex[2]);
			(contacts + id)->id0 = heightfieldid;
			(contacts + id)->id1 = rigidid;
			(contacts + id)->normal = pointnormal;// Vector3d(pointnormal[0], pointnormal[1], pointnormal[2]);
			(contacts + id)->mu = mu;

			//beInContact[tid] = 1;
		}
		

	}

	void HeightFieldBodyDetector::doCollision()
	{
		auto& allRigids = m_allBodies;
		//auto& allPBDBodies = solver->getCPUBody();

		if (this->var_Contacts.isEmpty())
			return;

		auto& contacts = this->var_Contacts.getValue();
		contacts.resize(0);

		int nmaxNum = 0;
		for (auto& prigid : allRigids)
		{
			auto triangleset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
			if (!triangleset)
				continue;
			DeviceArray<Vector3f>& vertices = triangleset->getPoints();
			nmaxNum += vertices.size();

		}
		if (contacts.capability() < nmaxNum)
		{
			contacts.reserve(nmaxNum);
		}

		//cudaError_t err0 = cudaGetLastError();
		int nContacts = 0;
		m_counter.resize(5);
		for (int i = 0; i < allRigids.size(); ++i)//auto& prigid : allRigids)
		{

			auto prigid = allRigids[i];
			auto triangleset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
			if (!triangleset)
				continue;
			DeviceArray<Vector3f>& vertices = triangleset->getPoints();

			//m_beInContact.resize(vertices.size());
			//m_beInContact.reset();
			m_counter.reset();

			RigidInteractionInfo rigidinfo;
			rigidinfo.position = prigid->getGlobalR();
			rigidinfo.rotation = prigid->getGlobalQ();
			rigidinfo.linearVelocity = prigid->getLinearVelocity();
			rigidinfo.angularVelocity = prigid->getAngularVelocity();

			// detect collision.
			float contactMu = prigid->getMu();
			//float threashold = rigidinfo.linearVelocity.norm() * dt;
			int bdim = 512;
			int gdim = cudaGridSize(vertices.size(), bdim);
			HFBDetect_detectContactPairPerRigidPoint << <gdim, bdim >> > (contacts.begin() + nContacts,
				m_counter.begin(),
				*m_land, vertices, rigidinfo,
				-1, i, contactMu, var_Threshold.getValue()
				);


			int curnum = 0;
			cudaMemcpy(&curnum, m_counter.begin(), sizeof(int), cudaMemcpyDeviceToHost);

			//cudaError_t err = cudaGetLastError();
			nContacts += curnum;
		}

		contacts.resize(nContacts);

		//solver->setContactJoints(m_contacts, m_nContacts);

	}

	void OrientedBodyDetector::doCollision()
	{
		if (m_transformedObbs.size() != m_obbs.size())
			m_transformedObbs.resize(m_obbs.size());

		// Update transformed obb position and orientation info.
		for (int i = 0; i < m_obbs.size(); ++i)
		{
			if (m_obbs[i] && m_allBodies[i])
			{
				// Generate transformed obb.
				if (!(m_transformedObbs[i]))
					m_transformedObbs[i] = std::make_shared<TOrientedBox3D<float>>();

				std::shared_ptr<TOrientedBox3D<float>> obb = m_obbs[i];
				std::shared_ptr<TOrientedBox3D<float>> tobb = m_transformedObbs[i];
				RigidBody2_ptr prigid = m_allBodies[i];

				// Compute transformed obb info.
				tobb->center = obb->center + prigid->getGlobalR();
				tobb->extent = obb->extent;
				tobb->u = prigid->getGlobalQ().rotate(obb->u);
				tobb->v = prigid->getGlobalQ().rotate(obb->v);
				tobb->w = prigid->getGlobalQ().rotate(obb->w);
			}
		}


		auto& contacts = this->var_Contacts.getValue();
		contacts.resize(0);
		contacts.reserve(m_obbs.size() * m_obbs.size());
		m_hostContacts.resize(0);
		m_hostContacts.reserve(contacts.capability());
		
		// Brute-force contact detection.
		for (int i = 0; i < m_obbs.size(); ++i)
		{
			if (!(m_transformedObbs[i]) || !(m_allBodies[i]))
				continue;

			for (int j = i + 1; j < m_obbs.size(); ++j)
			{
				if (!(m_transformedObbs[j]) || !(m_allBodies[j]))
					continue;
				bool collide = (m_allBodies[i]->getCollisionFilterGroup() & m_allBodies[j]->getCollisionFilterMask());
				collide = collide && (m_allBodies[j]->getCollisionFilterGroup() & m_allBodies[i]->getCollisionFilterMask());
				if (!collide)
					continue;

				// 
				Vector3f inNormal;
				Vector3f inPos0, inPos1;
				float inDis = 0;
				bool beIntersect = m_transformedObbs[i]->point_intersect(*(m_transformedObbs[j]), 
					inNormal, inDis, inPos0, inPos1);

				if (beIntersect)
				{
					m_hostContacts.resize(m_hostContacts.size() + 1);
					auto& curContact = m_hostContacts[m_hostContacts.size() - 1];
					curContact.id0 = m_allBodies[i]->getId();
					curContact.id1 = m_allBodies[j]->getId();
					curContact.normal = -Vector3d(inNormal[0], inNormal[1], inNormal[2]);
					curContact.point0 = Vector3d(inPos0[0], inPos0[1], inPos0[2]);
					curContact.point1 = Vector3d(inPos1[0], inPos1[1], inPos1[2]);
					curContact.mu = (m_allBodies[i]->getMu(), m_allBodies[j]->getMu()) / 2.0;
				}
			}
		}
		contacts.resize(m_hostContacts.size());
		if (m_hostContacts.size() > 0)
		{
			Function1Pt::copy(contacts, m_hostContacts);
		}
	}


}