//#include "Dynamics/RigidBody/PBDRigid/PointTriContactDetector.h"
//#include "Core/Utility/Function1Pt.h"
//
//
//#include <iostream>
//#include <Windows.h>
//
//
//namespace PhysIKA
//{
//	__device__ bool _PTCD_PointinTriangle(const Vector3f& A, const Vector3f& B, const Vector3f& C, const Vector3f& P)
//	{
//		Vector3f v0 = C - A;
//		Vector3f v1 = B - A;
//		Vector3f v2 = P - A;
//
//		float dot00 = v0.dot(v0);
//		float dot01 = v0.dot(v1);
//		float dot02 = v0.dot(v2);
//		float dot11 = v1.dot(v1);
//		float dot12 = v1.dot(v2);
//
//		float inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);
//
//		float u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
//		if (u < 0 || u > 1) // if u out of range, return directly
//		{
//			return false;
//		}
//
//		float v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
//		if (v < 0 || v > 1) // if v out of range, return directly
//		{
//			return false;
//		}
//
//		return u + v <= 1;
//	}
//
//	__global__ void _PTCD_computeContact(
//		ContactInfo<double>* contacts,
//		DeviceArray<Vector3f> points,
//		DeviceArray<Vector3f> triPoints,
//		DeviceArray<TopologyModule::Triangle> tri,
//		NeighborList<int> nbList,
//		int* counter, int id0, int id1, float mu0, float mu1,
//		float maxdis,
//		int maxContact
//	)
//	{
//		int pid = threadIdx.x + blockIdx.x * blockDim.x;
//		if (pid >= points.size()) return;
//		Vector3f p = points[pid];
//
//		int nbSize = nbList.getNeighborSize(pid);
//
//		//if(nbSize>0)
//			//printf("%d,  %d\n", pid, nbSize);
//
//		for (int i = 0; i < nbSize; ++i)
//		{
//			int triId = nbList.getElement(pid, i);
//			TopologyModule::Triangle curTri = tri[triId];
//
//			Vector3f pTri = triPoints[curTri[0]];
//			Vector3f n = (triPoints[curTri[1]] - pTri).cross(triPoints[curTri[2]] - pTri);
//			n.normalize();
//
//			float d = -((p - pTri).dot(n));
//			if (d < 0 || d>maxdis)
//				continue;
//
//
//			pTri = p + n * d;
//
//			if (!_PTCD_PointinTriangle(triPoints[curTri[0]], triPoints[curTri[1]], triPoints[curTri[2]], pTri))
//				continue;
//
//			//printf(" %d  %d,  %lf %lf %lf\n", pid, triId, p[0], p[1], p[2]);
//
//			int contactId = atomicAdd(counter, 1);
//			if (contactId >= maxContact) return;
//
//			contacts[contactId].id1 = id0;
//			contacts[contactId].id0 = id1;
//			contacts[contactId].mu = (mu0 + mu1) / 2.0;
//			contacts[contactId].normal = Vector3d(n[0], n[1], n[2]);
//			contacts[contactId].point1 = Vector3d(p[0], p[1], p[2]);
//			contacts[contactId].point0 = Vector3d(pTri[0], pTri[1], pTri[2]);
//		}
//	}
//
//
//	__global__ void _Detector_detectFromPointTriCondidate(
//		ContactInfo<double>* contacts,
//		DeviceDArray<PointTriContact<float>> condidates,
//		DeviceArray<Vector3f> transformnation,
//		DeviceArray<Quaternionf> rotation,
//		int* counter,
//		float maxdis,
//		int maxContact
//	)
//	{
//		int pid = threadIdx.x + blockIdx.x * blockDim.x;
//		if (pid >= condidates.size()) return;
//
//		PointTriContact<float>& ptc = condidates[pid];
//		int id0 = ptc.id0;
//		int id1 = ptc.id1;
//
//		Vector3f point = transformnation[id0] + rotation[id0].rotate(ptc.point0);
//		Vector3f p0 = transformnation[id1] + rotation[id1].rotate(ptc.triP0);
//		Vector3f p1 = transformnation[id1] + rotation[id1].rotate(ptc.triP1);
//		Vector3f p2 = transformnation[id1] + rotation[id1].rotate(ptc.triP2);
//
//		Vector3f n = (p1 - p0).cross(p2 - p0);
//		n.normalize();
//
//		float d = -((point - p0).dot(n));
//		if (d < 0 || d>maxdis)
//			return;
//
//		Vector3f cp1  = point + n * d;
//
//		if (!_PTCD_PointinTriangle(p0, p1, p2, cp1))
//			return;
//
//		int cid = atomicAdd(counter, 1);
//		if (cid >= maxContact) return;
//
//		contacts[cid].id1 = id0;
//		contacts[cid].id0 = id1;
//		contacts[cid].mu = ptc.mu;
//		contacts[cid].normal = Vector3d(n[0], n[1], n[2]);
//		contacts[cid].point1 = Vector3d(point[0], point[1], point[2]);
//		contacts[cid].point0 = Vector3d(cp1[0], cp1[1], cp1[2]);
//	}
//
//	__global__ void _PTCD_computeCandidate(
//		PointTriContact<float>* contacts,
//		DeviceArray<Vector3f> points,
//		DeviceArray<Vector3f> triPoints,
//		DeviceArray<TopologyModule::Triangle> tri,
//		DeviceArray<int> pointBelongTo,
//		NeighborList<int> nbList,
//		int* counter, float mu0, float mu1,
//		int maxContact
//	)
//	{
//		int pid = threadIdx.x + blockIdx.x * blockDim.x;
//		if (pid >= points.size()) return;
//
//		int nbSize = nbList.getNeighborSize(pid);
//		for (int i = 0; i < nbSize; ++i)
//		{
//			int triId = nbList.getElement(pid, i);
//			TopologyModule::Triangle curTri = tri[triId];
//
//			int cid = atomicAdd(counter, 1);
//			if (cid >= maxContact) return;
//
//			PointTriContact<float>& ptc= contacts[cid];
//			ptc.id0 = pid;
//			ptc.id1 = pointBelongTo[curTri[0]];
//			ptc.mu = (mu0 + mu1) / 2.0;
//			ptc.point0 = points[pid];
//			ptc.triP0 = triPoints[curTri[0]];
//			ptc.triP1 = triPoints[curTri[1]];
//			ptc.triP2 = triPoints[curTri[2]];
//		}
//
//	}
//
//
//	//__global__ void _Detector_setupObjectAABB(
//	//	DeviceArray<AABB> aabb,
//	//	DeviceArray<Vector3f> pos,
//	//	DeviceArray<float> radius,
//	//	float relaxation = 0.0f
//	//)
//	//{
//	//	int pid = threadIdx.x + blockIdx.x * blockDim.x;
//	//	if (pid >= pos.size()) return;
//
//	//	AABB box;
//	//	Vector3f p = pos[pid];
//	//	float r = radius[pid];
//	//	box.v0 = p - r- relaxation;
//	//	box.v1 = p + r + relaxation;
//	//	aabb[pid] = box;
//	//}
//
//	__global__ void _Detector_getContactPairs(
//		DeviceDArray<ContactPair> contPair,
//		NeighborList<int> nbList,
//		int* counter,
//		int maxContact
//	)
//	{
//		int tid = threadIdx.x + blockIdx.x * blockDim.x;
//		if (tid >= nbList.size()) return;
//
//		int num = nbList.getNeighborSize(tid);
//		if (num <= 0) return;
//
//		for (int i = 0; i < num; ++i)
//		{
//			int nbid = nbList.getElement(tid, i);
//
//			if (tid < nbid)
//			{
//				int contIdx = atomicAdd(counter, 1);
//				if (contIdx >= maxContact) return;
//
//				ContactPair cp;
//				cp.id[0] = tid;
//				cp.id[1] = nbid;
//				contPair[contIdx] = cp;
//			}
//		}
//	}
//
//	__global__ void _PTCD_updatePointsPose(
//		DeviceArray<Vector3f> points,
//		DeviceArray<Vector3f> oriPoints,
//		DeviceArray<int> pointBelongTo,
//		DeviceArray<Vector3f> trans,
//		DeviceArray<Quaternionf> rot
//	)
//	{
//		int tid = threadIdx.x + blockIdx.x * blockDim.x;
//		if (tid >= oriPoints.size())return;
//
//		int objid = pointBelongTo[tid];
//		points[tid] = trans[objid] + rot[objid].rotate(oriPoints[tid]);
//	}
//
//
//	__global__ void _Detector_computePointDis(
//		DeviceArray<float> dis,
//		DeviceArray<Vector3f> points
//	)
//	{
//		int tid = threadIdx.x + blockIdx.x * blockDim.x;
//		if (tid >= points.size())return;
//		dis[tid] = points[tid].norm();
//	}
//
//	PointTriContactDetector::PointTriContactDetector()
//	{
//		m_triQuery = std::make_shared< NeighborTriangleQuery<DataType3f>>();
//		m_broadPhaseCD = std::make_shared< CollisionDetectionBroadPhase<DataType3f>>();
//		m_reductionf = std::make_shared< Reduction<float>>();
//
//		m_triQuery->outNeighborhood()->connect(&m_neighborhood);
//		m_broadNeighbor.connect(m_broadPhaseCD->outContactList());
//
//		m_counter.resize(1);
//
//	}
//
//	PointTriContactDetector::~PointTriContactDetector()
//	{
//		m_counter.release();
//		m_rigidRadius.release();
//		m_rigidAABB.release();
//		m_triPos.release();
//		m_triPosHost.release();
//		m_contactPairs.release();
//		m_contactPairsHost.release();
//		m_contactInfos.release();
//	}
//
//	void PointTriContactDetector::initialize()
//	{
//		this->initRigidRadius();
//	}
//
//	int PointTriContactDetector::doBroadPhaseDetect(Real dt)
//	{
//		//if (!m_pRigids || m_pRigids->size() <= 0) return 0;
//
//		int num = 0;
//		num += this->detectPointTriCondidate(m_pointTriCondidate, cp.id[0], cp.id[1], dt);
//		num += this->detectPointTriCondidate(m_pointTriCondidate, cp.id[1], cp.id[0], dt);
//
//
//		return num;
//	}
//
//	int PointTriContactDetector::doNarrowDetect(Real dt)
//	{
//		int num = 0;;
//		m_contactInfos.resize(0);
//		num = this->detectFromPointTriCondidate(m_contactInfos);
//
//		return m_contactInfos.size();
//	}
//
//	int PointTriContactDetector::contectDetection(DeviceDArray<ContactInfo<double>>& contacts, int i, int j, Real dt, int begin)
//	{
//		if (!m_pRigids || m_pRigids->size() <= 0) return 0;
//
//		RigidBody2_ptr body0 = (*m_pRigids)[i];
//		RigidBody2_ptr body1 = (*m_pRigids)[j];
//
//
//		auto triangleset1 = TypeInfo::cast<TriangleSet<DataType3f>>(body1->getTopologyModule());
//		if (!triangleset1)
//			return 0;
//		DeviceArray<Vector3f>& vertices1 = triangleset1->getPoints();
//		auto triangles1 = triangleset1->getTriangles();
//
//		// Set input vertices points.
//		m_rigidPoints[i].connect(m_triQuery->inPosition());
//
//		// Set input triangle points.
//		m_rigidPoints[j].connect(m_triQuery->inTriangleVertex());
//		m_triQuery->inTriangleIndex()->setElementCount(triangles1->size());
//		auto& inTriIndices = m_triQuery->inTriangleIndex()->getValue();
//		Function1Pt::copy(inTriIndices, *(triangles1));
//
//		// Set detection radius.
//		float vel0 = body0->getLinearVelocity().norm();
//		float vel1 = body1->getLinearVelocity().norm();
//		float radius = 0.01+(vel0 + vel1)*dt * 20;
//		m_triQuery->inRadius()->getValue() = radius;
//
//		// Find contact (point, triangle) pair
//		m_triQuery->compute();
//
//		// Compute contact infomation.
//		m_counter.reset();
//		contacts.reserve(contacts.size() + m_maxContactPerPair);
//		cuExecute(m_rigidPoints[i].getValue().size(), _PTCD_computeContact,
//			contacts.begin() + contacts.size(),
//			m_rigidPoints[i].getValue(),
//			m_rigidPoints[j].getValue(), *triangles1,
//			m_neighborhood.getValue(),
//			m_counter.begin(),
//			i, j, body0->getMu(), body1->getMu(),
//			radius,
//			m_maxContactPerPair
//		);
//
//		int nCont = 0;
//		cudaMemcpy(&nCont, m_counter.begin(), sizeof(int), cudaMemcpyDeviceToHost);
//
//		contacts.resize(contacts.size() + nCont);
//
//		//if (nCont > 0)
//		//{
//
//		//	HostArray<Vector3f> hostpoints1;
//		//	hostpoints1.resize(m_rigidPoints[j].getValue().size());
//		//	Function1Pt::copy(hostpoints1, m_rigidPoints[j].getValue());
//
//		//	HostArray<ContactInfo<double>> hostContact;
//		//	hostContact.resize(contacts.size());
//		//	cudaMemcpy(hostContact.begin(), contacts.begin(), sizeof(ContactInfo<double>)*contacts.size(),
//		//		cudaMemcpyDeviceToHost);
//
//		//	hostContact.release();
//		//	hostpoints1.release();
//		//}
//		return nCont;
//	}
//
//
//	int PointTriContactDetector::detectPointTriCondidate(DeviceDArray<PointTriContact<float>>& contacts, int i, int j, Real dt, int begin)
//	{
//		this->updateTriPoint();
//
//		// Set input vertices points.
//		m_inPoints.connect(m_triQuery->inPosition());
//
//		// Set input triangle points.
//		m_globalMeshPoints.connect(m_triQuery->inTriangleVertex());
//		m_triVerIndices.connect(m_triQuery->inTriangleIndex());
//
//
//		// Find contact (point, triangle) pair
//		m_triQuery->compute();
//
//		// Compute contact infomation.
//		m_counter.reset();
//		contacts.reserve(m_maxContactPerPair);
//		cuExecute(m_inPoints.getValue().size(), _PTCD_computeCandidate,
//			contacts.begin(),
//			m_inPoints.getValue(),
//			m_globalMeshPoints.getValue(), m_triVerIndices.getValue(),
//			m_meshBelongTo.getValue(),
//			m_neighborhood.getValue(),
//			m_counter.begin(),
//			body0->getMu(), body1->getMu(),
//			m_maxContactPerPair
//		);
//
//		int nCont = 0;
//		cudaMemcpy(&nCont, m_counter.begin(), sizeof(int), cudaMemcpyDeviceToHost);
//
//		contacts.resize(nCont);
//		return nCont;
//	}
//
//	int PointTriContactDetector::detectFromPointTriCondidate(DeviceDArray<ContactInfo<double>>& contacts)
//	{
//		int maxCon = m_pointTriCondidate.size();
//		if (maxCon <= 0)
//			return 0;
//
//		this->_updateRigidPosInfo();
//		this->_updateRigidRotInfo();
//
//		float radius = m_detectionExt;
//
//		contacts.reserve(maxCon);
//		m_counter.reset();
//
//		cuExecute(maxCon, _Detector_detectFromPointTriCondidate,
//			contacts.begin(),
//			m_pointTriCondidate,
//			m_triPos,
//			m_triRot,
//			m_counter.begin(),
//			radius,
//			maxCon
//		);
//
//		int nCont = 0;
//		cudaMemcpy(&nCont, m_counter.begin(), sizeof(int), cudaMemcpyDeviceToHost);
//		contacts.resize(nCont);
//
//		//HostDArray<PointTriContact<float>> hostCondidate;
//		//hostCondidate.resize(m_pointTriCondidate.size());
//		//Function1Pt::copy(hostCondidate, m_pointTriCondidate);
//
//		//HostDArray<ContactInfo<double>> hostContact;
//		//hostContact.resize(contacts.size());
//		//Function1Pt::copy(hostContact, contacts);
//
//		//hostCondidate.release();
//		//hostContact.release();
//
//		return nCont;
//	}
//
//
//	void PointTriContactDetector::updateTriPoint()
//	{
//		if (m_globalMeshPoints.getElementCount() != m_localMeshPoints.getElementCount())
//			m_globalMeshPoints.setElementCount(m_localMeshPoints.getElementCount());
//
//		cuExecute(m_localMeshPoints.getElementCount(), _PTCD_updatePointsPose,
//			m_globalMeshPoints.getValue(),
//			m_globalMeshPoints.getValue(),
//			m_meshBelongTo.getValue(),
//			m_triPos, m_triRot
//		);
//
//	}
//
//	void PointTriContactDetector::initRigidRadius()
//	{
//		if (!m_pRigids || m_pRigids->size()<=0) return;
//		DeviceArray<float> dis;
//		HostArray<float> hostRadius;
//		hostRadius.resize(m_pRigids->size());
//		m_rigidRadius.resize(m_pRigids->size());
//
//		for (int i = 0; i < m_pRigids->size(); ++i)
//		{
//			auto prigid = (*m_pRigids)[i];
//			if (!prigid) continue;
//
//			auto triangleset1 = TypeInfo::cast<TriangleSet<DataType3f>>(prigid->getTopologyModule());
//			if (!triangleset1) continue;
//
//			dis.resize(triangleset1->getPointSize());
//			cuExecute(dis.size(), _Detector_computePointDis,
//				dis, triangleset1->getPoints()
//			);
//
//			hostRadius[i] = m_reductionf->maximum(dis.begin(), dis.size());
//		}
//
//		Function1Pt::copy(m_rigidRadius, hostRadius);
//		hostRadius.release();
//	}
//
//	void PointTriContactDetector::_updateRigidRotInfo()
//	{
//		if (!m_pRigids || m_pRigids->size() <= 0) return;
//
//		m_triRotHost.resize(m_pRigids->size());
//		for (int i = 0; i < m_pRigids->size(); ++i)
//		{
//			auto prigid = (*m_pRigids)[i];
//			if (prigid)
//			{
//				m_triRotHost[i] = prigid->getGlobalQ();
//			}
//			else
//			{
//				m_triRotHost[i] = Quaternionf();
//			}
//		}
//		m_triRot.resize(m_pRigids->size());
//		Function1Pt::copy(m_triRot, m_triRotHost);
//	}
//
//	void PointTriContactDetector::_updateRigidPosInfo()
//	{
//		if (!m_pRigids || m_pRigids->size() <= 0) return;
//
//		m_triPosHost.resize(m_pRigids->size());
//		for (int i = 0; i < m_pRigids->size(); ++i)
//		{
//			auto prigid = (*m_pRigids)[i];
//			if (prigid)
//			{
//				m_triPosHost[i] = prigid->getGlobalR();
//			}
//			else
//			{
//				m_triPosHost[i] = Vector3f();
//			}
//		}
//		m_triPos.resize(m_pRigids->size());
//		Function1Pt::copy(m_triPos, m_triPosHost);
//	}
//
//
//}