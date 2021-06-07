#include "Dynamics/RigidBody/PBDRigid/HeightFieldPBDInteractionNode.h"
#include "Framework/Framework/ModuleTopology.h"
//#include "Dynamics/Sand/swe/types.h"
//#include "Dynamics/RigidBody/RigidUtil.h"
//#include "Dynamics/RigidBody/RigidTimeIntegrationModule.h"
//#include "Dynamics/RigidBody/RKIntegrator.h"
#include <device_functions.h>

#include <iostream>

#include "thrust/reduce.h"
#include "thrust/execution_policy.h"
#include "thrust/device_vector.h"

#include "Dynamics/RigidBody/ContactInfo.h"

namespace PhysIKA
{
	HeightFieldPBDInteractionNode::~HeightFieldPBDInteractionNode()
	{
	}

	bool HeightFieldPBDInteractionNode::initialize()
	{
		m_solver->initialize();
		if (m_rigidContactDetector)
		{
			m_rigidContactDetector->varThreshold()->setValue(0.0001);
			m_rigidContactDetector->varContacts()->setValue(DeviceDArray<ContactInfo<double>>());
		}

		return true;
	}

	//bool HeightFieldPBDInteractionNode::initialize()
	//{
	//	return true;
	//}

	

	void HeightFieldPBDInteractionNode::setSize(int nx, int ny)
	{
		m_nx = nx;
		m_ny = ny;

		m_heightField.resize(nx, ny);
	}

	void HeightFieldPBDInteractionNode::setSize(int nx, int ny, float dx, float dy)
	{
		this->setSize(nx, ny);
		m_heightField.setSpace(dx, dy);
	}



	__global__ void detectContactPairPerRigidPoint(
		ContactInfo<double>* contacts,
		int* counter,
		DeviceHeightField1d grid,
		DeviceArray<Vector3f> vertices,
		RigidInteractionInfo rigidinfo,
		int heightfieldid, int rigidid, float mu, 
		float boundaryThreashold)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < vertices.size())
		{


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
				(contacts + id)->normal = Vector3d(pointnormal[0], pointnormal[1], pointnormal[2]);
				(contacts + id)->mu = mu;
				 
			}
		}
	}


	void HeightFieldPBDInteractionNode::advance(Real dt)
	{
		//CTimer timer;
		//timer.start();

		// Todo: Solve PBD.
		m_solver->advance(dt);
		
		//timer.stop();
		//double elapTime = timer.getElapsedTime();
		//printf("Solver time:  %lf \n", elapTime);
	}


	void HeightFieldPBDInteractionNode::contactDetection(PBDSolver* solver, Real dt)
	{

		auto& allRigids = solver->getRigidBodys();
		auto& allPBDBodies = solver->getCPUBody();

		int nmaxNum = 0;
		for (auto& prigid : allRigids)
		{
			auto triangleset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
			if (!triangleset)
				continue;
			DeviceArray<Vector3f>& vertices = triangleset->getPoints();
			nmaxNum += vertices.size();
		}
		if (m_contacts.size() < nmaxNum)
		{
			m_contacts.resize(nmaxNum);
		}

		//cudaError_t err0 = cudaGetLastError();
		m_nContacts = 0;
		nContactsi.resize(10);
		for (int i = 0; i < allRigids.size(); ++i)//auto& prigid : allRigids)
		{
			cudaMemset(nContactsi.begin(), 0, sizeof(int));

			auto prigid = allRigids[i];
			auto triangleset = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
			if (!triangleset)
				continue;
			DeviceArray<Vector3f>& vertices = triangleset->getPoints();

			RigidInteractionInfo rigidinfo;
			rigidinfo.position = prigid->getGlobalR();
			rigidinfo.rotation = prigid->getGlobalQ();
			rigidinfo.linearVelocity = prigid->getLinearVelocity();
			rigidinfo.angularVelocity = prigid->getAngularVelocity();

			// detect collision.
			float contactMu = allPBDBodies[i].mu;
			float threashold = rigidinfo.linearVelocity.norm() * dt;
			int bdim = 512;
			int gdim = cudaGridSize(vertices.size(), bdim);
			detectContactPairPerRigidPoint << <gdim, bdim >> > (m_contacts.begin() + m_nContacts,
				nContactsi.begin(),
				m_heightField, vertices, rigidinfo,
				-1, i, contactMu, threashold
				);

			int curnum = 0;
			//Function1Pt::copy(&curnum, nContactsi);
			cudaMemcpy(&curnum, nContactsi.begin(), sizeof(int), cudaMemcpyDeviceToHost);

			//cudaError_t err = cudaGetLastError();
			m_nContacts += curnum;
		}
		

		// Detect contact points between rigid bodies.
		if (m_rigidContactDetector)
		//if(false)
		{
			m_rigidContactDetector->doCollision();

			auto& contactArr = m_rigidContactDetector->varContacts()->getValue();
			if (contactArr.size() > 0)
			{
				m_contacts.resize(m_nContacts + contactArr.size());
				cudaMemcpy(&(m_contacts[m_nContacts]), contactArr.begin(),
					sizeof(ContactInfo<double>)*contactArr.size(), cudaMemcpyDeviceToDevice);

				m_nContacts += contactArr.size();
			}
		}

		
		// Todo: Update contact information to PBD joints.
		solver->setContactJoints(m_contacts, m_nContacts);
	}

	//__global__ void calculateDepth(DeviceGrid2Df grid,
	//	DeviceArray<ContactPointX> gridDepth,
	//	DeviceArray<TopologyModule::Triangle> faces,
	//	DeviceArray<Vector3f> vertices,
	//	RigidInteractionInfo rigidinfo)
	//{
	//	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	//	if (tid >= faces.size())
	//	{
	//		return;
	//	}

	//	//printf("%d  \n", tid);

	//	auto face = faces[tid];
	//	Vector3f p0 = rigidinfo.rotation.rotate(vertices[face[0]]) + rigidinfo.position;
	//	Vector3f p1 = rigidinfo.rotation.rotate(vertices[face[1]]) + rigidinfo.position;
	//	Vector3f p2 = rigidinfo.rotation.rotate(vertices[face[2]]) + rigidinfo.position;
	//	Vector3f faceNormal = (p1 - p0).cross(p2 - p0).normalize();

	//	float xmin = p0[0], ymin = p0[1], zmin = p0[2];
	//	float xmax = p0[0], ymax = p0[1], zmax = p0[2];
	//	xmin = p1[0] < xmin ? p1[0] : xmin;	ymin = p1[1] < ymin ? p1[1] : ymin;	zmin = p1[2] < zmin ? p1[2] : zmin;
	//	xmax = p1[0] > xmax ? p1[0] : xmax;	ymax = p1[1] > ymax ? p1[1] : ymax;	zmax = p1[2] > zmax ? p1[2] : zmax;
	//	xmin = p2[0] < xmin ? p2[0] : xmin;	ymin = p2[1] < ymin ? p2[1] : ymin;	zmin = p2[2] < zmin ? p2[2] : zmin;
	//	xmax = p2[0] > xmax ? p2[0] : xmax;	ymax = p2[1] > ymax ? p2[1] : ymax;	zmax = p2[2] > zmax ? p2[2] : zmax;

	//	// Find potential intersection grids.
	//	int2 gmin = grid.gridIndex(Vector3f(xmin, ymin, zmin));
	//	int2 gmax = grid.gridIndex(Vector3f(xmax, ymax, zmax));



	//	// For each grid, find intersection.
	//	for (int i = gmin.x; i <= gmax.x; ++i)
	//	{
	//		for (int j = gmin.y; j <= gmax.y; ++j)
	//		{
	//			Vector3f hitPoint;
	//			bool intersect = grid.onTopofGrid(i, j,
	//				p0, p1, p2, hitPoint);


	//			if (intersect)
	//			{
	//				float depth = grid(i, j) - hitPoint[1];
	//				if (depth >= 0)
	//				{

	//					ContactPointX cp;
	//					cp.depth = depth;
	//					cp.point = make_float3(hitPoint[0], hitPoint[1], hitPoint[2]);
	//					cp.normal = make_float3(faceNormal[0], faceNormal[1], faceNormal[2]);
	//					gridDepth[j*grid.Nx() + i] = cp;
	//				}
	//			}
	//		}
	//	}
	//}



	//float _contactPointRelV(const SystemMotionState& mState, const ContactPointX& cPoint, int id)
	//{
	//	Vector3f relp = Vector3f(cPoint.point.x, cPoint.point.y, cPoint.point.z) - mState.globalPosition[id];

	//	Transform3d<float> movetobody(mState.globalPosition[id], Quaternion<float>(0, 0, 0, 1));
	//	SpatialVector<float> vel = movetobody.transformM(mState.globalVelocity[id]);
	//	Vector3f linv(vel[3], vel[4], vel[5]);
	//	Vector3f angv(vel[0], vel[1], vel[2]);
	//	Vector3f newv = angv.cross(relp) + linv;
	//	return -newv.dot(Vector3f(cPoint.normal.x, cPoint.normal.y, cPoint.normal.z));
	//}




}