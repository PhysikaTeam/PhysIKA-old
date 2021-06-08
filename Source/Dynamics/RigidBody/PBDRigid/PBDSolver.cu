#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"
#include "Core/Utility/Function1Pt.h"



#include <device_functions.h>

#include <iostream>
#include <Windows.h>


namespace PhysIKA
{
	//template<typename TReal>
	PBDSolver::PBDSolver() :
		m_numSubstep(10),
		m_useGPU(false)
	{
	}

	//template<typename TReal>
	bool PBDSolver::initialize()
	{
		cudaDeviceSynchronize();
		cudaError_t err = cudaGetLastError();

		m_nPermanentBodies = m_rigids.size();
		m_nBodies = m_nPermanentBodies;
		m_nPermanentJoints = m_CPUJoints.size();
		m_nJoints = m_nPermanentJoints;

		

		int i = 0;
		for (auto prigid : m_rigids)
		{
			_toPBDBody(prigid, this->m_CPUBodys[i++]);
		}

		cudaDeviceSynchronize();
		err = cudaGetLastError();
		if (m_useGPU)
		{
			_initGPUData();
		}
		_updateBodyPointerInJoint();

		return true;
	}

	//template<typename TReal>
	void PBDSolver::advance(Real dt)
	{
		m_detectionTime = 0.0;
		m_timer.start();

		if (m_bodyDataDirty)
		{
			_onRigidDataDirty();
			m_bodyDataDirty = false;
		}

		if (m_broadPhaseDetection)
			m_broadPhaseDetection(this, dt);



		double subdt = dt / m_numSubstep;
		for (int i = 0; i < m_numSubstep; ++i)
		{
			// debug
			//Vector3d tmpvel = m_CPUBodys[0].linVelocity;
			//printf("  substep: %d,   body 0 vel: %lf %lf %lf\n", substepCount, tmpvel[0], tmpvel[1], tmpvel[2]);
			//++substepCount;

			this->forwardSubStep(subdt);
		}


		m_bodyDataDirty = false;
		m_jontDataDirty = false;

		m_timer.stop();
		double elapTime = m_timer.getElapsedTime();
		m_totalTime += elapTime;
		m_totalFrame += 1;
		double averageTime = m_totalTime / m_totalFrame;
		printf("PBD solve time: without detect %lf,  with detect %lf ,    Average Time:  %lf\n", elapTime - m_detectionTime,  elapTime, averageTime);

	}

	void PBDSolver::integrateBody(Real dt)
	{
		if (!m_useGPU)
		{
			for (int i = 0; i < m_nBodies; ++i)
			{
				auto& body = m_CPUBodys[i];
				body.integrate(dt);
			}
		}
	}

	void PBDSolver::solveJoints(Real dt)
	{
		if (!m_useGPU)
		{
			for (int i = 0; i < m_nJoints; ++i)
			{
				auto& joint = m_CPUJoints[i];
				joint.solvePose(dt);
			}
		}
	}


	//template<typename TReal>
	void PBDSolver::CPUUpdate(Real dt)
	{
		Real subtimestep = dt / m_numSubstep;

		for (int stepi = 0; stepi < m_numSubstep; ++stepi)
		{



			//for (int ji = 0; ji < 10; ++ji)
			//{
			//	for (int i = 0; i < m_nJoints; ++i)
			//	{
			//		auto& joint = m_CPUJoints[i];
			//		joint.testGhostInitPose(subtimestep);
			//	}
			//}

			for (int i = 0; i < m_nBodies; ++i)
			{
				auto& body = m_CPUBodys[i];
				body.integrate(subtimestep);
			}

			synFromBodiedToRigid();
			this->doCustomUpdate(dt);

			if (m_narrowPhaseDetection)
			{
				m_narrowPhaseDetection(this, subtimestep);
			}




			for (int i = 0; i < m_nJoints; ++i)
			{
				auto& joint = m_CPUJoints[i];
				joint.solvePose(subtimestep);
			}


			for (int i = 0; i < m_nBodies; ++i)
			{
				auto& body = m_CPUBodys[i];
				body.updateVelocity(subtimestep);
			}

			for (int i = 0; i < m_nJoints; ++i)
			{
				auto& joint = m_CPUJoints[i];
				joint.solveVelocity(subtimestep);
			}

			synFromBodiedToRigid();
		}
	}



	void PBDSolver::doCustomUpdate(Real dt)
	{
		for (int i = 0; i < m_selfupdate.size(); ++i)
		{
			if (m_selfupdate[i])
			{
				m_selfupdate[i](dt);
			}
		}

		if (m_selfupdate.size() > 0)
		{
			this->updateRigidToPBDBody();
		}
	}

	//template<typename TReal>
	int PBDSolver::addRigid(RigidBody2_ptr prigid)
	{
		if (prigid)
		{
			//this->addChild(prigid);
			m_rigids.push_back(prigid);

			PBDBodyInfo<double> pbdrigid;
			_toPBDBody(prigid, pbdrigid);
			int rid = this->addPBDRigid(pbdrigid);
			prigid->setId(rid);
			return rid;
		}
		return -1;
	}

	//template<typename TReal>
	int PBDSolver::addPBDRigid(const PBDBodyInfo<double>& pbdbody)
	{
		m_CPUBodys.push_back(pbdbody);
		
		m_nBodies = m_CPUBodys.size();
		return m_CPUBodys.size() - 1;
	}

	//template<typename TReal>
	int PBDSolver::addPBDJoint(const PBDJoint<double>& pbdjoint, int bodyid0, int bodyid1)
	{
		m_CPUJoints.push_back(pbdjoint);
		int id = m_CPUJoints.size() - 1;
		auto& joint = m_CPUJoints[id];
		joint.body0 = bodyid0 >= 0 ? &(m_CPUBodys[bodyid0]) : 0;
		joint.body1 = bodyid1 >= 0 ? &(m_CPUBodys[bodyid1]) : 0;
		joint.bodyId0 = bodyid0;
		joint.bodyId1 = bodyid1;
		
		m_nJoints = m_CPUJoints.size();
		m_nPermanentJoints = m_nJoints;
		return id;
	}

	

	//template<typename TReal>
	void PBDSolver::_toPBDBody(RigidBody2_ptr prigid, PBDBodyInfo<double>& pbdbody)
	{
		Vector3f pos = prigid->getGlobalR();
		Quaternionf rot = prigid->getGlobalQ();
		pbdbody.pose.position = Vector<double, 3>(pos[0], pos[1], pos[2]);
		pbdbody.pose.rotation = Quaternion<double>(rot.x(), rot.y(), rot.z(), rot.w());

		Vector3f linv = prigid->getLinearVelocity();
		Vector3f angv = prigid->getAngularVelocity();
		pbdbody.linVelocity = Vector<double, 3>(linv[0], linv[1], linv[2]);
		pbdbody.angVelocity = Vector<double, 3>(angv[0], angv[1], angv[2]);

		Inertia<float> inertia = prigid->getI();
		Vector3f I = inertia.getInertiaDiagonal();
		pbdbody.invMass = inertia.getMass() == 0.0 ? 0.0 : 1.0 / inertia.getMass();
		pbdbody.invInertia = Vector<double, 3>(
			I[0] == 0.0 ? 0.0 : 1.0 / I[0],
			I[1] == 0.0 ? 0.0 : 1.0 / I[1],
			I[2] == 0.0 ? 0.0 : 1.0 / I[2]);

		Vector3f extF = prigid->getExternalForce();
		Vector3f extT = prigid->getExternalTorque();
		pbdbody.externalForce = Vector<double, 3>(extF[0], extF[1], extF[2]);
		pbdbody.externalTorque = Vector<double, 3>(extT[0], extT[1], extT[2]);

		pbdbody.mu = prigid->getMu();
		pbdbody.rho = prigid->getRho();

		pbdbody.linDamping = prigid->getLinearDamping();
		pbdbody.angDamping = prigid->getAngularDamping();
	}

	//template<typename TReal>
	void PBDSolver::_fromPBDBody(const PBDBodyInfo<double>& pbdbody, RigidBody2_ptr prigid)
	{
		Vector3f gloP(pbdbody.pose.position[0], pbdbody.pose.position[1], pbdbody.pose.position[2]);
		Quaternionf gloQ(pbdbody.pose.rotation.x(), pbdbody.pose.rotation.y(), pbdbody.pose.rotation.z(), pbdbody.pose.rotation.w());
		prigid->setGlobalR(gloP);
		prigid->setGlobalQ(gloQ);

		Vector3f linV(pbdbody.linVelocity[0], pbdbody.linVelocity[1], pbdbody.linVelocity[2]);
		Vector3f angV(pbdbody.angVelocity[0], pbdbody.angVelocity[1], pbdbody.angVelocity[2]);
		prigid->setLinearVelocity(linV);
		prigid->setAngularVelocity(angV);
	}

	void PBDSolver::synFromBodiedToRigid()
	{
		if (m_useGPU && m_nPermanentBodies > 0)
		{
			cudaMemcpy(&(m_CPUBodys[0]), m_GPUBodys.begin(),
				sizeof(PBDBodyInfo<double>)*m_nPermanentBodies, cudaMemcpyDeviceToHost);
		}

		int i = 0;
		for (auto prigid : m_rigids)
		{
			auto& body = m_CPUBodys[i];
			_fromPBDBody(body, prigid);
			++i;

			//printf("  Pos:  %lf, %lf, %lf \n",
			//	body.pose.position[0], body.pose.position[1], body.pose.position[2]);
		}
	}

	
	
	__global__ void PBD_generateContactJoints(PBDJoint<double>* joints,
		//DeviceArray<PBDBodyInfo<double>> bodies,
		DeviceDArray<ContactInfo<double>> contacts, int njoint)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid < njoint)
		{
			PBDJoint<double>* joint = joints + tid;
			joint->bodyId0 = contacts[tid].id0;
			joint->bodyId1 = contacts[tid].id1;
			joint->body0 = 0;
			joint->body1 = 0;
			//if (joint->bodyId0 >= 0)
			//	joint->body0 = &(bodies[joint->bodyId0]);
			//if (joint->bodyId1 >= 0)
			//	joint->body1 = &(bodies[joint->bodyId1]);

			joint->globalPose0.position = contacts[tid].point0;
			joint->globalPose1.position = contacts[tid].point1;
			joint->updateLocalPoses();

			joint->mu = contacts[tid].mu;
			joint->normal = contacts[tid].normal;
			joint->beContact = true;
			joint->rotationXLimited = false;
			joint->rotationYLimited = false;
			joint->rotationZLimited = false;
			joint->positionLimited = true;
			joint->maxDistance = -1.0;
			joint->restitution = 0.0;
			joint->compliance = 0.000000;

			joint->active = true;
		}
	}

	__global__ void PBD_updateJointLocalPose(PBDJoint<double>* joints, int nJoints)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		if (tid < nJoints)
		{
			(joints + tid)->updateLocalPoses();
		}
	}
	

	void PBDSolver::setContactJoints(DeviceDArray<ContactInfo<double>>& contacts, int nContact)
	{
		m_nJoints = m_nPermanentJoints + nContact;

		if (nContact <= 0 || m_nJoints<=0)
			return;

		
		// Resize data.
		if (m_nJoints> 0 && m_GPUJoints.size() < m_nJoints)
		{
			m_GPUJoints.resize(m_nJoints);
			if (m_nPermanentJoints > 0)
			{
				cudaMemcpy(m_GPUJoints.begin(), &(m_CPUJoints[0]),
					sizeof(PBDJoint<double>)* m_nPermanentJoints, cudaMemcpyHostToDevice);
			}
		}
		

		// Build contact joints.
		int bdim = m_blockdim;
		int gdim = (nContact + bdim - 1) / bdim;
		PBD_generateContactJoints << <gdim, bdim >> > (m_GPUJoints.begin() + m_nPermanentJoints,
			/*m_GPUBodys, */contacts, nContact);
		//cuSynchronize();
		
		cuSynchronize();


		if (!m_useGPU)
		{
			// Copy joint information to host.
			if (m_CPUJoints.size() < m_nJoints)
			{
				m_CPUJoints.resize(m_nJoints);
				
			}
			cudaMemcpy(&(m_CPUJoints[0]), m_GPUJoints.begin(),
				sizeof(PBDJoint<double>)*m_nJoints, cudaMemcpyDeviceToHost);
		}
		_updateBodyPointerInJoint();

		//HostArray<PBDJoint<double>> hostjoints;
		//hostjoints.resize(m_nJoints);
		//Function1Pt::copy(hostjoints, m_GPUJoints);

		//hostjoints.release();

		if (m_useGPU)
		{
			PBD_updateJointLocalPose << <gdim, bdim >> > (m_GPUJoints.begin() + m_nPermanentJoints, nContact);
		}
		else
		{
			for (int i = 0; i < nContact; ++i)
			{
				m_CPUJoints[m_nPermanentJoints + i].updateLocalPoses();
			}
		}
	}



	/***
	* *******************   GPU solver code.   ***********************
	*/

	__global__ void initJoints(PBDJoint<double>* joints, int nJoints,
		PBDBodyInfo<double>* bodys)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;

		if (tid < nJoints)
		{
			PBDJoint<double>* joint = joints + tid;
			joint->body0 = joint->bodyId0 >= 0 ? bodys + joint->bodyId0 : 0;
			joint->body1 = joint->bodyId1 >= 0 ? bodys + joint->bodyId1 : 0;
		}
	}


	void PBDSolver::_updateBodyPointerInJoint()
	{
		if (m_useGPU && m_nJoints>0)
		{
			int blockdim = m_blockdim;
			int griddim = (m_nJoints + blockdim - 1) / blockdim;
			initJoints << <griddim, blockdim >> > (m_GPUJoints.begin(), m_nJoints,
				m_GPUBodys.begin());
			cuSynchronize();
		}
		else
		{
			for (int i = 0; i < m_nJoints; ++i)
			{
				PBDJoint<double>& joint = m_CPUJoints[i];
				joint.body0 = joint.bodyId0 >= 0 ? &(m_CPUBodys[joint.bodyId0]) : 0;
				joint.body1 = joint.bodyId1 >= 0 ? &(m_CPUBodys[joint.bodyId1]) : 0;
			}
		}
	}


	//template<typename TReal>
	void PBDSolver::_initGPUData()
	{
		int nbody = m_CPUBodys.size();
		int njoint = m_CPUJoints.size();


		if (njoint > 0)
		{
			m_GPUJoints.resize(njoint);
			Function1Pt::copy(m_GPUJoints, m_CPUJoints);
		}

		if (nbody > 0)
		{
			m_GPUBodys.resize(nbody);
			Function1Pt::copy(m_GPUBodys, m_CPUBodys);

			m_GPUPosChange.resize(nbody);
			m_GPURotChange.resize(nbody);
			m_GPULinvChange.resize(nbody);
			m_GPUAngvChange.resize(nbody);
			m_GPUConstraintCount.resize(nbody);
			m_GPUOmega.resize(nbody);
		}
	}

	void PBDSolver::_onRigidDataDirty()
	{
		int i = 0;
		for (auto prigid : m_rigids)
		{
			_toPBDBody(prigid, this->m_CPUBodys[i++]);
		}

		if (m_useGPU && m_nPermanentBodies>0)
		{
			cudaMemcpy(m_GPUBodys.begin(), &(m_CPUBodys[0]), 
				sizeof(PBDBodyInfo<double>)*m_nPermanentBodies, cudaMemcpyHostToDevice);
		}
	}

	void PBDSolver::updateRigidToPBDBody()
	{
		//m_nPermanentBodies = m_rigids.size();

		this->updateRigidToCPUBody();

		if (m_useGPU)
		{
			this->updateCPUToGPUBody();
		}
		
	}

	void PBDSolver::updateRigidToCPUBody()
	{
		for (int i = 0; i < m_rigids.size(); ++i)
		{
			_toPBDBody(m_rigids[i], m_CPUBodys[i]);
		}
	}

	void PBDSolver::updateRigidToGPUBody()
	{
		this->updateRigidToCPUBody();
		this->updateCPUToGPUBody();
	}

	void PBDSolver::updateCPUToGPUBody()
	{
		if (m_nPermanentBodies > 0)
		{
			cudaMemcpy(m_GPUBodys.begin(), &(m_CPUBodys[0]),
				sizeof(PBDBodyInfo<double>)*m_nPermanentBodies,
				cudaMemcpyHostToDevice);
		}
	}

	void PBDSolver::updateGPUToCPUBody()
	{
		if (m_nPermanentBodies > 0)
		{
			cudaMemcpy(&(m_CPUBodys[0]), m_GPUBodys.begin(),
				sizeof(PBDBodyInfo<double>)*m_nPermanentBodies,
				cudaMemcpyDeviceToHost);
		}
	}

	



	__device__ void updateGlobalPose(BodyPose<double>* ppose0, BodyPose<double>* ppose1, PBDJoint<double>& joint)
	{
		joint.globalPose0 = joint.localPose0;
		if (ppose0)
			ppose0->tranformPose(joint.globalPose0);
		joint.globalPose1 = joint.localPose1;
		if (ppose1)
			ppose1->tranformPose(joint.globalPose1);
	}

	COMM_FUNC Vector3d getQuatAxis0(const Quaterniond& q)
	{
		double x2 = q.x() * 2.0;
		double w2 = q.w() * 2.0;
		return Vector3d((q.w() * w2) - 1.0 + q.x() * x2, (q.z() * w2) + q.y() * x2, (-q.y() * w2) + q.z() * x2);
	}
	COMM_FUNC Vector3d getQuatAxis1(const Quaterniond& q)
	{
		double y2 = q.y() * 2.0;
		double w2 = q.w() * 2.0;
		return Vector3d((-q.z() * w2) + q.x() * y2, (q.w() * w2) - 1.0 + q.y() * y2, (q.x() * w2) + q.z() * y2);
	}
	COMM_FUNC Vector3d _getQuatAxis2(const Quaterniond& q)
	{
		double z2 = q.z() * 2.0;
		double w2 = q.w() * 2.0;
		return Vector3d((q.y() * w2) + q.x() * z2, (-q.x() * w2) + q.y() * z2, (q.w() * w2) - 1.0 + q.z() * z2);
	}


	__device__ void positionAdd(Vector3d& p0, const Vector3d& p1)
	{
		atomicAdd(&p0[0], p1[0]);
		atomicAdd(&p0[1], p1[1]);
		atomicAdd(&p0[2], p1[2]);
	}
	__device__ void rotationAdd(Quaterniond& q0, const Quaterniond& q1)
	{
		//double v=0.0;
		//atomicAdd(&v, 1);
		atomicAdd(&q0[0], q1.x());
		atomicAdd(&q0[1], q1.y());
		atomicAdd(&q0[2], q1.z());
		atomicAdd(&q0[3], q1.w());

	}

	__device__ double bodyInvMass(const PBDBodyInfo<double>* pbody, 
		const BodyPose<double>& pose,
		const Vector3d& normal, const Vector3d* pos)
	{
		if (pbody == 0)
			return 0.0;

		Vector3d n(normal);
		double w = 0;
		if (pos)
		{
			n = (*pos - pose.position).cross(normal);
			w = pbody->invMass;
		}

		pose.invRotate(n);
		w += n[0] * n[0] * pbody->invInertia[0] + 
			n[1] * n[1] * pbody->invInertia[1] + 
			n[2] * n[2] * pbody->invInertia[2];
		if(pbody->nContacts>0)
			w *= pbody->nContacts;
		return w;
	}


	__device__ void integrateRotation(BodyPose<double>& pose, 
		const Vector3d& rot, double scale = 1.0)
	{
		double phi = rot.norm();
		scale *= 0.5;
		Quaterniond dq(rot[0] * scale, rot[1] * scale, rot[2] * scale, 0.0);
		pose.rotation = pose.rotation + dq * pose.rotation;
		pose.rotation.normalize();
	}

	
	/**
	* @brief Apply velocity level correction.
	*/
	__device__ void applyVelocityCorrection(
		Vector3d& linv, Vector3d& angv,
		const BodyPose<double>& pose,
		const PBDBodyInfo<double>* pbody,
		const Vector3d& corr, const Vector3d* pos
	)
	{
		Vector3d dp(corr);

		// Linear velocity correction.
		if (pos)
		{
			linv += corr * pbody->invMass * pbody->nContacts;
			dp = ((*pos) - pose.position).cross(corr);
		}

		// Angular velocity correction.
		pose.invRotate(dp);					// to local frame.
		dp *= pbody->invInertia * pbody->nContacts;			// angular velocity in local frame.
		pose.rotate(dp);					// to world frame.

		//printf(" Body inv inertia: %lf %lf %lf dp: %lf %lf %lf\n", 
		//	pbody->invInertia[0], pbody->invInertia[1], pbody->invInertia[2],
		//	dp[0], dp[1], dp[2]);
		angv += dp;

	}

	/**
	* @brief Apply body position and rotation correction.
	*/
	__device__ void applyBodyCorrection(
		BodyPose<double>& pose,
		const PBDBodyInfo<double>* pbody,
		const Vector3d& corr, const Vector3d* pos)
	{
		Vector3d dp(corr);

		// Position correction.
		if (pos)
		{
			pose.position += corr * pbody->invMass * pbody->nContacts;
			dp = ((*pos) - pose.position).cross(corr);
		}

		// rotation correction.
		pose.invRotate(dp);					// to local frame.
		dp *= pbody->invInertia * pbody->nContacts;			// angular velocity in local frame.
		pose.rotate(dp);					// to world frame.
		integrateRotation(pose, dp);
	}

	

	__device__ double applyBodyPairCorrection(
		BodyPose<double>& pose0, BodyPose<double>& pose1,
		const PBDBodyInfo<double>* body0, const PBDBodyInfo<double>* body1,
		const PBDJoint<double>& joint, 
		const Vector3d& corr, 
		double dt, bool bePositionCorr,double maxlambda = -1.0) 
	{
		double C = corr.norm();
		if (C <= EPSILON)
			return 0.0;
		Vector3d n(corr);
		n.normalize();

		double w0 = bodyInvMass(body0, pose0, n, bePositionCorr? &joint.globalPose0.position: 0);
		double w1 = bodyInvMass(body1, pose1, n, bePositionCorr? &joint.globalPose1.position: 0);

		//if (w0 <= 0 && w1 <= 0)
		//{
		//	printf("ERROR:  w0: %lf,  w1: %lf Corr:%lf %lf %lf\n", w0, w1, corr[0], corr[1], corr[2]);

		//	//printf("ERROR:  w0: %lf,  w1: %lf body0:%d, body1:%d\n", w0, w1, body0?body0->nContacts:0, body1 ? body1->nContacts : 0);

		//	//printf("ERROR:  w0: %lf,  w1: %lf body0:%d, body1:%d\n", w0, w1, body0 == 0 ? 0 : 1, body1 == 0 ? 0 : 1);
		//}

		double lambda = C * (dt*dt) / ((w0 + w1)*(dt*dt)/* + joint.compliance*/);
		lambda = maxlambda >= 0 && lambda > maxlambda ? maxlambda : lambda;
		n *= lambda;


		if (body0)
		{
			applyBodyCorrection(pose0, body0, n, 
				bePositionCorr ? &joint.globalPose0.position : 0);
		}
		if (body1)
		{
			n *= -1;
			applyBodyCorrection(pose1, body1, n,
				bePositionCorr ? &joint.globalPose1.position : 0);
		}
		return lambda;
	}


	/**
	* *@brief Apply body pair velocity level correction.
	*/
	__device__ double applyPairVelocityCorrection(
		Vector3d& linv0, Vector3d& angv0, Vector3d& linv1, Vector3d& angv1,
		const PBDBodyInfo<double>* body0, const PBDBodyInfo<double>* body1,
		const PBDJoint<double>& joint,
		const Vector3d& corr,
		double dt, bool bePositionCorr, double maxlambda = -1.0)
	{
		double C = corr.norm();
		if (C <= 0)
			return 0.0;
		Vector3d n(corr);
		n.normalize();

		double w0 = 0.0;
		if (body0)
			//w0 = body0->getPBDInverseMass(n, bePositionCorr ? joint.globalPose0.position : Vector3d(0, 0, 0));
			w0 = bodyInvMass(body0, body0->pose, n, bePositionCorr ? &joint.globalPose0.position : 0);
		double w1 = 0.0; 
		if (body1)
			//w1 = body1->getPBDInverseMass(n, bePositionCorr ? joint.globalPose1.position : Vector3d(0, 0, 0));
			w1 = bodyInvMass(body1, body1->pose, n, bePositionCorr ? &joint.globalPose1.position : 0);

		double lambda = C * (dt*dt) / ((w0 + w1)*(dt*dt) + joint.compliance);
		lambda = maxlambda >= 0 && lambda > maxlambda ? maxlambda : lambda;

		n *= lambda;

		if (body0)
		{
			applyVelocityCorrection(linv0, angv0,
				body0->pose, body0, n,
				bePositionCorr ? &joint.globalPose0.position : 0);
		}
		if (body1)
		{
			n *= -1;
			applyVelocityCorrection(linv1, angv1,
				body1->pose, body1, n,
				bePositionCorr ? &joint.globalPose1.position : 0);
		}
		return lambda;
	}

	__device__ double limitAngle(
		BodyPose<double>& pose0, BodyPose<double>& pose1,
		PBDBodyInfo<double>* body0, PBDBodyInfo<double>* body1,
		const PBDJoint<double>& joint,
		const Vector3d& n, const Vector3d& a, const Vector3d& b,
		double minAngle, double maxAngle, double dt, double maxCorr = CUDART_PI)
	{
		Vector3d axb = a.cross(b);
		double axbdot = axb.dot(n);

		double phi = asin(min(1.0, max(-1.0, axbdot)));
		if (a.dot(b) < 0)
			phi = CUDART_PI - phi;
		if (phi > CUDART_PI)
			phi -= 2.0*CUDART_PI;
		if (phi < -CUDART_PI)
			phi += 2.0*CUDART_PI;

		if (phi < minAngle || phi >maxAngle)
		{
			phi = minAngle > phi ? minAngle : phi;
			phi = maxAngle < phi ? maxAngle : phi;

			Quaterniond q(n, phi);
			Vector3d omega = q.rotate(a);
			omega = omega.cross(b);

			phi = omega.norm();
			if (phi > maxCorr)
				omega *= maxCorr / phi;
			return applyBodyPairCorrection(pose0, pose1, body0, body1, joint,
				omega, dt, false);
		}
		return 0;
	}

	//__device__ void limitPosition()

	__global__ void PBDForward(PBDBodyInfo<double>* bodys, int nBodys,
		PBDJoint<double>* joints, int nJoints,
		Vector3d * dx, Quaterniond* dq, int* nConstraint,
		int nStep, double dt, double ome)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		double subdt = dt / nStep;


		for (int i = 0; i < nStep; ++i)
		{
			// set dx and dq
			if (tid < nBodys)
			{
				*(dx + tid) = Vector3d(0, 0, 0);
				*(dq + tid) = Quaterniond(0, 0, 0, 0);
				*(nConstraint + tid) = 0;
			}
			// ** integrate
			if (tid < nBodys)
			{
				(bodys+tid)->integrate(subdt);
			}
			__syncthreads();

			// ** solve joint.
			if (tid < nJoints)
			{
				PBDJoint<double>* joint = joints + tid;
				BodyPose<double> pose0;
				if ((joints + tid)->body0)
					pose0 = (joints + tid)->body0->pose;
				BodyPose<double> pose1;
				if ((joints + tid)->body1)
					pose1 = (joints + tid)->body1->pose;

				// X rotation.
				if ((joints + tid)->rotationXLimited)
				{
					updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
						(joints + tid)->body1 ? &pose1 : 0,
						*(joints + tid));
					//Vector3d a0 = getQuatAxis0((joints + tid)->globalPose0.rotation);
					//Vector3d a1 = getQuatAxis0((joints + tid)->globalPose1.rotation);
					Vector3d a0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
					Vector3d a1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
					a1 -= a0 * (a1.dot(a0));
					a1.normalize();
					a0 = getQuatAxis0((joints + tid)->globalPose0.rotation);
					Vector3d n = a0.cross(a1).normalize();
					limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
						n, a0, a1, 
						(joints+tid)->minAngleX, (joints+tid)->maxAngleX, subdt);
				}

				// Y rotation.
				if ((joints + tid)->rotationYLimited)
				{
					updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
						(joints + tid)->body1 ? &pose1 : 0,
						*(joints + tid));
					Vector3d b0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
					Vector3d b1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
					Vector3d n = b0.cross(b1);
					n.normalize();

					limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
						n, b0, b1,
						(joints + tid)->minAngleY, (joints + tid)->maxAngleY, subdt);
				}

				// Z rotation.
				if ((joints + tid)->rotationZLimited)
				{
					updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
						(joints + tid)->body1 ? &pose1 : 0,
						*(joints + tid));
					//Vector3d b0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
					//Vector3d b1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
					Vector3d n = getQuatAxis1((joints + tid)->globalPose0.rotation).cross(
						getQuatAxis1((joints + tid)->globalPose1.rotation)
					).normalize();

					Vector3d c0 = _getQuatAxis2((joints + tid)->globalPose0.rotation);
					c0 = (c0 - n * (c0.dot(n)));
					c0.normalize();
					Vector3d c1 = _getQuatAxis2((joints + tid)->globalPose1.rotation);
					c1 = (c1 - n * (c1.dot(n)));
					c1.normalize();

					
					limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
						n, c0, c1,
						(joints + tid)->minAngleZ, (joints + tid)->maxAngleZ, subdt);
				}


				// Position constraint.
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));

				Vector3d pcorr = (joints + tid)->globalPose1.position - (joints + tid)->globalPose0.position;
				applyBodyPairCorrection(pose0, pose1,
					(joints + tid)->body0, (joints + tid)->body1,
					*(joints + tid),
					pcorr, subdt, true);


				// Update postion change.
				if ((joints + tid)->body0)
				{

					PBDBodyInfo<double>* body = (joints+tid)->body0;
					Vector3d dpos = pose0.position - body->pose.position;
					Quaterniond drot = pose0.rotation - body->pose.rotation;
					int bodyid = (joints + tid)->bodyId0;
					positionAdd(*(dx + bodyid), dpos);
					rotationAdd(*(dq + bodyid), drot);
					atomicAdd((nConstraint + bodyid), 1);
				}
				if ((joints + tid)->body1)
				{
					PBDBodyInfo<double>* body = (joints + tid)->body1;
					Vector3d dpos = pose1.position - body->pose.position;
					Quaterniond drot = pose1.rotation - body->pose.rotation;
					int bodyid = (joints + tid)->bodyId1;
					positionAdd(*(dx + bodyid), dpos);
					rotationAdd(*(dq + bodyid), drot);
					atomicAdd((nConstraint + bodyid), 1);
				}
			}
			__syncthreads();

			// Update SOR postion and update velocity.
			if (tid < nBodys)
			{
				

				PBDBodyInfo<double>* body = (bodys + tid);
				double alpha = ome / *(nConstraint + tid); 
				alpha =  alpha > 1.0 ? 1.0 : alpha;

				


				body->pose.position += (*(dx + tid)) * alpha;
				body->pose.rotation += (*(dq + tid)) * alpha;
				body->pose.rotation.normalize();

				body->updateVelocity(subdt);

			}
			__syncthreads();

		}
	}



	/***
	* @brief Update contact joint activation state.
	*/
	__global__ void PBD_updateContactActivation(PBDJoint<double>* joints, int nJoints, double threashold)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nJoints && (joints + tid)->beContact)
		{
			PBDJoint<double>* joint = joints + tid;
			updateGlobalPose(joint->body0 ? &(joint->body0->pose) : 0,
				joint->body1 ? &(joint->body1->pose) : 0,
				*joint);
			Vector3d relp = joint->globalPose1.position - joint->globalPose0.position;
			if (relp.dot(joint->normal) > threashold)
			{
				joint->active = false;
			}
			else
			{
				joint->active = true;
			}
		}
	}

	/**
	* @brief Integrate body velocity and position. And body prepare delta data.
	*/
	__global__ void PBD_integrateAndPrepareBody(PBDBodyInfo<double>* bodys, 
		Vector3d* dx, Quaterniond* dq, Vector3d* dlinv, Vector3d* dangv, int* nConstraint,
		int nbodys, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nbodys)
		{
			(bodys + tid)->integrate(dt);

			*(dx + tid) = Vector3d(0, 0, 0);
			*(dq + tid) = Quaterniond(0, 0, 0, 0);
			*(dlinv + tid) = Vector3d(0, 0, 0);
			*(dangv + tid) = Vector3d(0, 0, 0);
			*(nConstraint + tid) = 0;
			(bodys + tid)->nContacts = 0;
		}
	}

	__global__ void PBD_prepareJointInfo(PBDJoint<double>* joints, int njoint)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < njoint)
		{
			PBDJoint<double>* joint = joints + tid;
			joint->normLambda = 0.0;
			joint->tangLambda0 = 0.0;
			joint->tangLambda1 = 0.0;
		}
	}


	__global__ void PBD_integrateForce(PBDBodyInfo<double>* bodys,
		int nbodys, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nbodys)
		{
			(bodys + tid)->integrateForce(dt);
		}
	}


	__global__ void PBD_integrateVelocity(PBDBodyInfo<double>* bodys,
		Vector3d* dx, Quaterniond* dq, Vector3d* dlinv, Vector3d* dangv, int* nConstraint,
		int nbodys, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nbodys)
		{
			(bodys + tid)->integrateVelocity(dt);

			*(dx + tid) = Vector3d(0, 0, 0);
			*(dq + tid) = Quaterniond(0, 0, 0, 0);
			*(dlinv + tid) = Vector3d(0, 0, 0);
			*(dangv + tid) = Vector3d(0, 0, 0);
			*(nConstraint + tid) = 0;
			(bodys + tid)->nContacts = 0;
		}
	}

	__global__ void PBD_updateConstraintNumber(PBDJoint<double>* joints,
		int nJoints)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nJoints && (joints + tid)->active)
		{
			PBDJoint<double>* joint = joints + tid;
			if (joint->body0)
				atomicAdd(&(joint->body0->nContacts), 1);
			if (joint->body1)
				atomicAdd(&(joint->body1->nContacts), 1);
		}
	}

	/**
	* @brief Solve PBD joints.
	*/
	__global__ void PBD_solveJoints(PBDJoint<double>* joints, int nJoints, 
		Vector3d* dx, Quaterniond* dq, int* nConstraint, int nbodys, 
		double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nJoints && (joints+tid)->active)
		{
			
			PBDJoint<double>* joint = joints + tid;
			BodyPose<double> pose0;
			if ((joints + tid)->body0)
				pose0 = (joints + tid)->body0->pose;
			BodyPose<double> pose1;
			if ((joints + tid)->body1)
				pose1 = (joints + tid)->body1->pose;

			// X rotation.
			if ((joints + tid)->rotationXLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				Vector3d a0 = getQuatAxis0((joints + tid)->globalPose0.rotation);
				Vector3d a1 = getQuatAxis0((joints + tid)->globalPose1.rotation);
				//Vector3d a0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
				//Vector3d a1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
				////a1 -= a0 * (a1.dot(a0));
				//a1 = a0.cross(a1);
				////a1.normalize();
				//a0 = getQuatAxis0((joints + tid)->globalPose0.rotation);
				Vector3d n = a0.cross(a1);
				if (n.norm() > 0 && a1.norm()>0)
				{
					n.normalize();
					a1.normalize();
					limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
						n, a0, a1,
						(joints + tid)->minAngleX, (joints + tid)->maxAngleX, dt);
				}
			}

			// Y rotation.
			if ((joints + tid)->rotationYLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				Vector3d b0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
				Vector3d b1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
				Vector3d n = b0.cross(b1);
				n.normalize();

				limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
					n, b0, b1,
					(joints + tid)->minAngleY, (joints + tid)->maxAngleY, dt);
			}

			// Z rotation.
			if ((joints + tid)->rotationZLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				//Vector3d b0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
				//Vector3d b1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
				Vector3d n = getQuatAxis1((joints + tid)->globalPose0.rotation).cross(
					getQuatAxis1((joints + tid)->globalPose1.rotation)
				).normalize();

				Vector3d c0 = _getQuatAxis2((joints + tid)->globalPose0.rotation);
				c0 = (c0 - n * (c0.dot(n)));
				c0.normalize();
				Vector3d c1 = _getQuatAxis2((joints + tid)->globalPose1.rotation);
				c1 = (c1 - n * (c1.dot(n)));
				c1.normalize();

				limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
					n, c0, c1,
					(joints + tid)->minAngleZ, (joints + tid)->maxAngleZ, dt);
			}

			// Position constraint.
			if(joint->positionLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));

				Vector3d pcorr = (joints + tid)->globalPose1.position - (joints + tid)->globalPose0.position;
				Vector3d tmpv = (joints+tid)->beContact?
					(joints+tid)->normal: - getQuatAxis1((joints + tid)->globalPose0.rotation);
				double dn = pcorr.dot(tmpv);
				if ((joints + tid)->beContact)
				{
					Vector3d relv = (joints + tid)->getJointVelocity();
					double pn = (1 + joint->restitution) * tmpv.dot(relv) * dt;

					////if (dn < pn)
					//{
					//	printf("Dn, Pn:  %lf %lf \n", dn, pn);
					//}

					dn = dn < pn ? dn : pn;
					//dn = pn;
				}
				pcorr = tmpv * dn;
				double lambda = 0.0;

				// limite normal position.
				if (dn < 0
					|| ((joints + tid)->maxDistance >= 0 && dn > (joints + tid)->maxDistance))
				{
					if((joints + tid)->maxDistance >= 0 && dn > (joints + tid)->maxDistance)
						pcorr *= (dn - (joints + tid)->maxDistance) / dn;

					lambda = applyBodyPairCorrection(pose0, pose1,
						(joints + tid)->body0, (joints + tid)->body1,
						*(joints + tid),
						pcorr, dt, true);
					joint->normLambda = lambda /*/ dt*/;
				}

				// limite tangent position
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				pcorr = joint->globalPose1.position - joint->globalPose0.position;
				dn = pcorr.dot(tmpv);
				pcorr -= tmpv * dn;
				if ((joints + tid)->beContact)
				{
					lambda *= joint->mu;

					pcorr = Vector3d(0.0, 0.0, 0.0);
					if (joint->body1)
					{
						pcorr = joint->localPose1.position;
						joint->body1->prevPose.transform(pcorr);
						pcorr = joint->globalPose1.position - pcorr;
					}
					tmpv = Vector3d(0.0, 0.0, 0.0);
					if (joint->body0)
					{
						tmpv = joint->localPose0.position;
						joint->body0->prevPose.transform(tmpv);
						tmpv = joint->globalPose0.position - tmpv;
					}
					pcorr -= tmpv;
					pcorr = pcorr - joint->normal * pcorr.dot(joint->normal);
					//pcorr = -pcorr;
					//printf("Friction solve\n");
					
				}
				else 
				{
					lambda = -1.0;
				}

				Vector3d deb_p1 = joint->localPose1.position;
				pose1.transform(deb_p1);
				Vector3d deb_pc = pcorr;

				lambda = applyBodyPairCorrection(pose0, pose1,
					(joints + tid)->body0, (joints + tid)->body1,
					*(joints + tid),
					pcorr, dt, true, lambda);
				
				// update normal velocity
				if ((joint)->beContact)
				{
					//Vector3d deb_p1_ = joint->localPose1.position;
					//pose1.transform(deb_p1_);
					//deb_p1 = deb_p1_ - deb_p1;
					//if (joint->bodyId1 == 2 /*|| joint->bodyId0 == 2*/)
					//	printf("Friction solve: %d, %d,  fric lambda, %lf; mu, %lf;  corr, %lf %lf %lf;  dp, %lf %lf %lf\n", 
					//		joint->bodyId0, joint->bodyId1, lambda/dt, joint->mu, //joint->normLambda,
					//		deb_pc[0], deb_pc[1], deb_pc[2],
					//		deb_p1[0], deb_p1[1], deb_p1[2]);

					tmpv = joint->getJointVelocity();
					joint->relVn = tmpv.dot(joint->normal);
				}
			}

			// Update postion change.
			if ((joints + tid)->body0)
			{

				PBDBodyInfo<double>* body = (joints + tid)->body0;
				Vector3d dpos = pose0.position - body->pose.position;
				Quaterniond drot = pose0.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId0;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
				atomicAdd((nConstraint + bodyid), 1);
				//atomicAdd(&(body->nContacts), 1);
			}
			if ((joints + tid)->body1)
			{
				PBDBodyInfo<double>* body = (joints + tid)->body1;
				Vector3d dpos = pose1.position - body->pose.position;
				Quaterniond drot = pose1.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId1;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
				atomicAdd((nConstraint + bodyid), 1);
				//atomicAdd(&(body->nContacts), 1);

			}
		}
	}


	/**
	* @brief Solve PBD joints.
	*/
	__global__ void PBD_solveConstraintJoints(PBDJoint<double>* joints, int nJoints,
		Vector3d* dx, Quaterniond* dq, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nJoints && (joints + tid)->active && !((joints + tid)->beContact))
		{
			PBDJoint<double>* joint = joints + tid;
			BodyPose<double> pose0;
			if ((joints + tid)->body0)
				pose0 = (joints + tid)->body0->pose;
			BodyPose<double> pose1;
			if ((joints + tid)->body1)
				pose1 = (joints + tid)->body1->pose;

			// debug
			double deb_lambda_x = -1;
			double deb_lambda_y = -1;
			double deb_lambda_z = -1;
			double deb_lambda_n = -1;
			double deb_lambda_t = -1;


			// X rotation.
			if ((joints + tid)->rotationXLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				Vector3d a0 = getQuatAxis0((joints + tid)->globalPose0.rotation);
				Vector3d a1 = getQuatAxis0((joints + tid)->globalPose1.rotation);
				Vector3d n = a0.cross(a1);
				if (n.norm() > 0 && a1.norm() > 0 &&a0.norm()>0)
				{
					n.normalize();
					a1.normalize();
					a0.normalize();
					deb_lambda_x = limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
						n, a0, a1,
						(joints + tid)->minAngleX, (joints + tid)->maxAngleX, dt);
				}
			}

			// Y rotation.
			if ((joints + tid)->rotationYLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				Vector3d b0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
				Vector3d b1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
				Vector3d n = b0.cross(b1);
				n.normalize();
				b0.normalize();
				b1.normalize();

				deb_lambda_y = limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
					n, b0, b1,
					(joints + tid)->minAngleY, (joints + tid)->maxAngleY, dt);
			}

			// Z rotation.
			if ((joints + tid)->rotationZLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				Vector3d b0 = getQuatAxis1((joints + tid)->globalPose0.rotation);
				Vector3d b1 = getQuatAxis1((joints + tid)->globalPose1.rotation);
				//Vector3d n = getQuatAxis1((joints + tid)->globalPose0.rotation).cross(
				//	getQuatAxis1((joints + tid)->globalPose1.rotation)
				//).normalize();
				Vector3d n = (b0 + b1);
				if (n.norm() > EPSILON)
				{
					n = n.normalize();

					Vector3d c0 = _getQuatAxis2((joints + tid)->globalPose0.rotation);
					c0 = (c0 - n * (c0.dot(n)));
					c0.normalize();
					Vector3d c1 = _getQuatAxis2((joints + tid)->globalPose1.rotation);
					c1 = (c1 - n * (c1.dot(n)));
					c1.normalize();

					deb_lambda_z = limitAngle(pose0, pose1, (joints + tid)->body0, (joints + tid)->body1, *(joints + tid),
						n, c0, c1,
						(joints + tid)->minAngleZ, (joints + tid)->maxAngleZ, dt);
				}
			}

			// Position constraint.
			if (joint->positionLimited)
			{
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));

				Vector3d pcorr = joint->globalPose1.position - joint->globalPose0.position;
				Vector3d tmpv = -getQuatAxis1(joint->globalPose0.rotation);
				double dn = pcorr.dot(tmpv);
				pcorr = tmpv * dn;
				double lambda = 0.0;

				// limite normal position.
				if (dn < 0
					|| ((joints + tid)->maxDistance >= 0 && dn > (joints + tid)->maxDistance))
				{
					if ((joints + tid)->maxDistance >= 0 && dn > (joints + tid)->maxDistance)
						pcorr *= (dn - (joints + tid)->maxDistance) / dn;

					lambda = applyBodyPairCorrection(pose0, pose1,
						(joints + tid)->body0, (joints + tid)->body1,
						*(joints + tid),
						pcorr, dt, true);
					joint->normLambda = lambda / dt;
				}
				deb_lambda_n = lambda;

				// limite tangent position
				updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
					(joints + tid)->body1 ? &pose1 : 0,
					*(joints + tid));
				pcorr = joint->globalPose1.position - joint->globalPose0.position;
				dn = pcorr.dot(tmpv);
				pcorr -= tmpv * dn;
				

				Vector3d deb_p1 = joint->localPose1.position;
				pose1.transform(deb_p1);
				Vector3d deb_pc = pcorr;

				lambda = applyBodyPairCorrection(pose0, pose1,
					(joints + tid)->body0, (joints + tid)->body1,
					*(joints + tid),
					pcorr, dt, true);

				deb_lambda_t = lambda;
			}

			//printf("Lambdas: %lf %lf %lf %lf %lf\n", deb_lambda_x, deb_lambda_y, deb_lambda_z, deb_lambda_n, deb_lambda_t);

			// Update postion change.
			if ((joints + tid)->body0)
			{

				PBDBodyInfo<double>* body = (joints + tid)->body0;
				Vector3d dpos = pose0.position - body->pose.position;
				Quaterniond drot = pose0.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId0;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
			}
			if ((joints + tid)->body1)
			{
				PBDBodyInfo<double>* body = (joints + tid)->body1;
				Vector3d dpos = pose1.position - body->pose.position;
				Quaterniond drot = pose1.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId1;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
			}
		}
	}


	/**
	* @brief Solve contacts.
	*/
	__global__ void PBD_solveContactJoints(PBDJoint<double>* joints, int nJoints,
		Vector3d* dx, Quaterniond* dq, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nJoints && (joints + tid)->active && (joints + tid)->beContact)
		{

			PBDJoint<double>* joint = joints + tid;
			BodyPose<double> pose0;
			if (joint->body0)
				pose0 = joint->body0->pose;
			BodyPose<double> pose1;
			if (joint->body1)
				pose1 = joint->body1->pose;

			// ******* Constraint normal position *******
			// Position constraint.
			updateGlobalPose(joint->body0 ? &pose0 : 0,
				joint->body1 ? &pose1 : 0,
				*joint);

			Vector3d pcorr = joint->globalPose1.position - joint->globalPose0.position;
			Vector3d tmpv = joint->normal;
			double dn = pcorr.dot(tmpv);
			{
				Vector3d prevpos1 = joint->localPose1.position;
				if (joint->body1)
					joint->body1->prevPose.transform(prevpos1);
				Vector3d prevpos0 = joint->localPose0.position;
				if (joint->body0)
					joint->body0->prevPose.transform(prevpos0);
				double prevpn = tmpv.dot(prevpos1 - prevpos0);

				double pn = -(joint->restitution) * tmpv.dot(joint->getJointVelocity()) * dt + prevpn;

				dn -= pn < 0 ? 0 : pn;
				//dn = dn < pn ? dn : pn;
				//dn = pn;
			}

			pcorr = tmpv * dn;
			double lambda = 0.0;

			// Constraint normal position.
			if (dn < 0)
			{
				//if ((joints + tid)->maxDistance >= 0 && dn > (joints + tid)->maxDistance)
				//	pcorr *= (dn - (joints + tid)->maxDistance) / dn;

				lambda = applyBodyPairCorrection(pose0, pose1,
					(joints + tid)->body0, (joints + tid)->body1,
					*(joints + tid),
					pcorr, dt, true);
				joint->normLambda = lambda /*/ dt*/;

				lambda = abs(lambda * joint->mu);
				//printf("Mu:  %lf\n", joint->mu);
			}


			// ***** Constraint tangent position.  ******
			// Update global pose
			updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
				(joints + tid)->body1 ? &pose1 : 0,
				*(joints + tid));

			pcorr = Vector3d(0.0, 0.0, 0.0);
			if (joint->body1)
			{
				pcorr = joint->localPose1.position;
				joint->body1->prevPose.transform(pcorr);
				pcorr = joint->globalPose1.position - pcorr;
			}
			tmpv = Vector3d(0.0, 0.0, 0.0);
			if (joint->body0)
			{
				tmpv = joint->localPose0.position;
				joint->body0->prevPose.transform(tmpv);
				tmpv = joint->globalPose0.position - tmpv;
			}
			pcorr -= tmpv;
			pcorr = pcorr - joint->normal * pcorr.dot(joint->normal);

			// debug 
			double curmaxlambda = lambda;

			// Apply tangent position correction.
			lambda = applyBodyPairCorrection(pose0, pose1,
				(joints + tid)->body0, (joints + tid)->body1,
				*(joints + tid),
				pcorr, dt, true, lambda);

			//if (/*lambda >= curmaxlambda&&*/ curmaxlambda>0)
			//{
			//	printf("Fric up to max: %lf %lf, %d \n", lambda, curmaxlambda, lambda>= curmaxlambda?1:0);
			//}


			// Update postion change.
			if ((joints + tid)->body0)
			{

				PBDBodyInfo<double>* body = (joints + tid)->body0;
				Vector3d dpos = pose0.position - body->pose.position;
				Quaterniond drot = pose0.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId0;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
			}
			if ((joints + tid)->body1)
			{
				PBDBodyInfo<double>* body = (joints + tid)->body1;
				Vector3d dpos = pose1.position - body->pose.position;
				Quaterniond drot = pose1.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId1;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
			}
		}
	}

	__device__ void _addPose(BodyPose<double>& pose, const Vector3d&dx, const Quaterniond& dq, double alpha = 1.0)
	{
		pose.position += dx * alpha;
		pose.rotation += dq * alpha;
		pose.rotation.normalize();
	}

	/**
	* @brief Update body velocity using SOR update.
	*/
	__global__ void PBD_updatePosChangeAndVelocity(PBDBodyInfo<double>* bodys, int nbodys,
		Vector3d* dx, Quaterniond* dq, /*int* nConstraint,*/
		 double dt, double ome, bool updateVel=true, bool resetNConstraint=false)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nbodys )
		{
			PBDBodyInfo<double>* body = (bodys + tid);
			if (body &&body->nContacts>0)
			{
				double alpha = ome / (body->nContacts);// ome / *(nConstraint + tid);
				//alpha = alpha > 1.0 ? 1.0 : alpha;

				body->pose.position += (*(dx + tid)) * alpha;
				body->pose.rotation += (*(dq + tid)) * alpha;
				body->pose.rotation.normalize();

				//if (tid == 2)
				//{
				//	Vector3d dp = (*(dx + tid)) * alpha;
				//	printf("Pos change: %lf %lf %lf \n", dp[0], dp[1], dp[2]);
				//}
			}

			if (updateVel)
			{
				body->updateVelocity(dt);
			}
			if (resetNConstraint)
			{
				body->nContacts = 0;
			}

			//*(nConstraint + tid) = 0;
		}
	}

	/**
	* @brief Update body velocity according to position change.
	*/
	__global__ void PBD_updateVelocity(PBDBodyInfo<double>* bodys, int nbodys, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nbodys)
		{
			(bodys + tid)->updateVelocity(dt);
		}
	}

	/**
	* @brief Update body pose using SOR update.
	*/
	__global__ void PBD_updatePosChange(PBDBodyInfo<double>* bodys,
		Vector3d* dx, Quaterniond* dq, 
		int nbodys, double dt, double ome, bool resetNConstraint = false)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nbodys)
		{
			PBDBodyInfo<double>* body = (bodys + tid);
			if (body &&body->nContacts > 0)
			{
				//double alpha = ome[tid] / *(nConstraint + tid);
				//alpha = alpha > 1.0 ? 1.0 : alpha;
				double alpha = ome / (body->nContacts);// ome / *(nConstraint + tid);

				body->pose.position += (*(dx + tid)) * alpha;
				body->pose.rotation += (*(dq + tid)) * alpha;
				body->pose.rotation.normalize();
			}

			if (resetNConstraint)
			{
				body->nContacts = 0;
			}
		}

	}

	

	/**
	* @brief solve velocity constraints.
	*/
	__global__ void PBD_solveVelocity(PBDJoint<double>* joints, int nJoints,
		Vector3d* dlinv, Vector3d* dangv, /*int* nConstraint,*/ int nbodys,
		double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if(tid < nJoints && (joints + tid)->active)
		{
			PBDJoint<double>* joint = joints + tid;
			Vector3d linv0(0, 0, 0);
			Vector3d angv0(0, 0, 0);
			if (joint->body0)
			{
				linv0 = joint->body0->linVelocity;
				angv0 = joint->body0->angVelocity;
			}
			Vector3d linv1(0, 0, 0);
			Vector3d angv1(0, 0, 0);
			if (joint->body1)
			{
				linv1 = joint->body1->linVelocity;
				angv1 = joint->body1->angVelocity;
			}


			// Damp rotation velocity.
			Vector3d dv(0, 0, 0);
			if (joint->body0)
				dv -= joint->body0->angVelocity;
			if (joint->body1)
				dv += joint->body1->angVelocity;
			dv *= joint->angDamping *dt > 1.0 ? 1.0 : joint->angDamping*dt;
			applyPairVelocityCorrection(linv0, angv0, linv1, angv1,
				(joints + tid)->body0, (joints + tid)->body1,
				*(joints + tid),
				dv, dt, false);


			// Damp linear velocity.
			dv[0] = 0.0;	dv[1] = 0.0;	dv[2] = 0.0;
			if (joint->body0)
				dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
			if (joint->body1)
				dv += joint->body1->getVelocityAt(joint->globalPose1.position);
			dv *= joint->linDamping *dt > 1.0 ? 1.0 : joint->linDamping*dt;
			applyPairVelocityCorrection(linv0, angv0, linv1, angv1,
				(joints + tid)->body0, (joints + tid)->body1,
				*(joints + tid),
				dv, dt, true);

			//printf("%d:  angv0:  %lf %lf %lf\n", tid, angv0[0], angv0[1], angv0[2]);


			// Handle restitution.
			if (joint->beContact && joint->relVn<=0)
			{
				dv[0] = 0.0;	dv[1] = 0.0;	dv[2] = 0.0;
				if (joint->body0)
					dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
				if (joint->body1)
					dv += joint->body1->getVelocityAt(joint->globalPose1.position);
				double ev = - joint->restitution * joint->relVn;
				ev = dv.dot(joint->normal) - (ev > 2.0*dt*9.8 ? ev : 0);
				dv = joint->normal * ev;
				applyPairVelocityCorrection(linv0, angv0, linv1, angv1,
					(joints + tid)->body0, (joints + tid)->body1,
					*(joints + tid),
					dv, dt, true);
			}

			//printf("%d:  angv0:  %lf %lf %lf\n", tid, angv0[0], angv0[1], angv0[2]);
			//printf("%d:  angv0:  %lf %lf %lf\n", tid, angv1[0], angv1[1], angv1[2]);


			// Frictional velocitty.
			if (joint->beContact)
			{
				dv[0] = 0.0;	dv[1] = 0.0;	dv[2] = 0.0;
				if (joint->body0)
					dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
				if (joint->body1)
					dv += joint->body1->getVelocityAt(joint->globalPose1.position);

				Vector3d vt = dv - joint->normal * dv.dot(joint->normal);

				double ev = vt.norm();
				ev = (ev == 0 || ev < (joint->mu * joint->normLambda / dt)) ? 1.0 : (joint->mu * joint->normLambda / dt / ev);
				dv = vt * ev;
				applyPairVelocityCorrection(
					linv0, angv0, linv1, angv1,
					(joints + tid)->body0, (joints + tid)->body1,
					*(joints + tid),
					dv, dt, true);
			}

			// Update postion change.
			if (joint->body0)
			{
				Vector3d dlinvi = linv0 - joint->body0->linVelocity;
				Vector3d dangvi = angv0 - joint->body0->angVelocity;
				int bodyid = joint->bodyId0;
				positionAdd(*(dlinv + bodyid), dlinvi);
				positionAdd(*(dangv + bodyid), dangvi);
				//atomicAdd((nConstraint + bodyid), 1);
			}
			if ((joints + tid)->body1)
			{
				Vector3d dlinvi = linv1 - joint->body1->linVelocity;
				Vector3d dangvi = angv1 - joint->body1->angVelocity;
				int bodyid = joint->bodyId1;
				positionAdd(*(dlinv + bodyid), dlinvi);
				positionAdd(*(dangv + bodyid), dangvi);
				//atomicAdd((nConstraint + bodyid), 1);

			}
		}

	}

	/**
	* @brief solve velocity constraints.
	*/
	__global__ void PBD_solveCollisionVelocity(PBDJoint<double>* joints, int nJoints,
		Vector3d* dlinv, Vector3d* dangv,  int nbodys,
		double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= nJoints || !((joints + tid)->active)) return;


		PBDJoint<double>* joint = joints + tid;
		Vector3d linv0(0, 0, 0);
		Vector3d angv0(0, 0, 0);
		if (joint->body0)
		{
			linv0 = joint->body0->linVelocity;
			angv0 = joint->body0->angVelocity;
		}
		Vector3d linv1(0, 0, 0);
		Vector3d angv1(0, 0, 0);
		if (joint->body1)
		{
			linv1 = joint->body1->linVelocity;
			angv1 = joint->body1->angVelocity;
		}

		// Handle restitution.
		if (joint->beContact && joint->relVn <= 0)
		{

			updateGlobalPose(joint->body0 ? &(joint->body0->pose) : 0,
				joint->body1 ? & (joint->body1->pose) : 0,
				*joint);

			Vector3d dv;
			if (joint->body0)
				dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
			if (joint->body1)
				dv += joint->body1->getVelocityAt(joint->globalPose1.position);
			double ev = -joint->restitution * joint->relVn;
			ev = dv.dot(joint->normal) - (ev > 2.0*dt*9.8 ? ev : 0);
			dv = joint->normal * ev;
			applyPairVelocityCorrection(linv0, angv0, linv1, angv1,
				joint->body0, joint->body1,
				*joint,
				dv, dt, true);


			// Frictional velocitty.
			dv[0] = 0.0;	dv[1] = 0.0;	dv[2] = 0.0;
			if (joint->body0)
				dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
			if (joint->body1)
				dv += joint->body1->getVelocityAt(joint->globalPose1.position);

			Vector3d vt = dv - joint->normal * dv.dot(joint->normal);

			ev = vt.norm();
			ev = (ev == 0 || ev < (joint->mu * joint->normLambda / dt)) ? 1.0 : (joint->mu * joint->normLambda / dt / ev);
			dv = vt * ev;
			applyPairVelocityCorrection(
				linv0, angv0, linv1, angv1,
				joint->body0, joint->body1,
				*joint,
				dv, dt, true);
		}

		// Update postion change.
		if (joint->body0)
		{
			Vector3d dlinvi = linv0 - joint->body0->linVelocity;
			Vector3d dangvi = angv0 - joint->body0->angVelocity;
			int bodyid = joint->bodyId0;
			positionAdd(*(dlinv + bodyid), dlinvi);
			positionAdd(*(dangv + bodyid), dangvi);
		}
		if ((joints + tid)->body1)
		{
			Vector3d dlinvi = linv1 - joint->body1->linVelocity;
			Vector3d dangvi = angv1 - joint->body1->angVelocity;
			int bodyid = joint->bodyId1;
			positionAdd(*(dlinv + bodyid), dlinvi);
			positionAdd(*(dangv + bodyid), dangvi);
		}


	}


	/**
	* @brief Update body velocity change using SOR update.
	*/
	__global__ void PBD_updateVelocityChange(PBDBodyInfo<double>* bodys,
		Vector3d* dlinv, Vector3d* dangv, 
		int nbodys, double dt, double ome)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nbodys)
		{
			PBDBodyInfo<double>* body = (bodys + tid);
			//if (*(nConstraint + tid) > 0)
			if(body && body->nContacts>0)
			{
				double alpha = ome / body->nContacts;/// *(nConstraint + tid);
				alpha = alpha > 1.0 ? 1.0 : alpha;

				body->linVelocity += (*(dlinv + tid)) * alpha;
				body->angVelocity += (*(dangv + tid)) * alpha;
			}
		}
	}


	__global__ void PBD_solveCollisionPose(PBDJoint<double>* joints, int nJoints,
		Vector3d* dx, Quaterniond* dq, int* nConstraint, int nbodys,
		double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		
		if (tid < nJoints && (joints + tid)->active && (joints + tid)->beContact)
		{

			PBDJoint<double>* joint = joints + tid;
			BodyPose<double> pose0;
			if ((joints + tid)->body0)
				pose0 = (joints + tid)->body0->pose;
			BodyPose<double> pose1;
			if ((joints + tid)->body1)
				pose1 = (joints + tid)->body1->pose;

			updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
				(joints + tid)->body1 ? &pose1 : 0,
				*(joints + tid));

			Vector3d jv = joint->getJointVelocity();
			double jvn = jv.dot(joint->normal);
			joint->relVn = jvn;
			if (jvn >= 0)
				return;

			Vector3d pcorr = joint->normal * ((1.0 + joint->restitution)*jvn*dt);
			joint->normLambda = applyBodyPairCorrection(pose0, pose1,
				(joints + tid)->body0, (joints + tid)->body1,
				*(joints + tid),
				pcorr, dt, true);

			// Update postion change.
			if ((joints + tid)->body0)
			{

				PBDBodyInfo<double>* body = (joints + tid)->body0;
				Vector3d dpos = pose0.position - body->pose.position;
				Quaterniond drot = pose0.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId0;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
				atomicAdd((nConstraint + bodyid), 1);
			}
			if ((joints + tid)->body1)
			{
				PBDBodyInfo<double>* body = (joints + tid)->body1;
				Vector3d dpos = pose1.position - body->pose.position;
				Quaterniond drot = pose1.rotation - body->pose.rotation;
				int bodyid = (joints + tid)->bodyId1;
				positionAdd(*(dx + bodyid), dpos);
				rotationAdd(*(dq + bodyid), drot);
				atomicAdd((nConstraint + bodyid), 1);

			}
		}
	}

	__global__ void PBD_omegaCorrection(PBDJoint<double>* joints, int nJoints,
		Vector3d* dx, Quaterniond* dq, double * omega, int* nConstraint, int nbodys,
		double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid < nJoints && (joints + tid)->active && (joints + tid)->beContact)
		{
			PBDJoint<double>* joint = joints + tid;
			BodyPose<double> pose0;
			if ((joints + tid)->body0)
				pose0 = (joints + tid)->body0->pose;
			BodyPose<double> pose1;
			if ((joints + tid)->body1)
				pose1 = (joints + tid)->body1->pose;

			if (joint->body0)
			{
				int bodyid = joint->bodyId0;
				double alpha = 1.0 / *(nConstraint + bodyid);
				_addPose(pose0, *(dx + bodyid), *(dq + bodyid), alpha);
			}
			if (joint->body1)
			{
				int bodyid = joint->bodyId1;
				double alpha = 1.0 / *(nConstraint + bodyid);
				_addPose(pose0, *(dx + bodyid), *(dq + bodyid), alpha);
			}

			double curd = joint->normal.dot(joint->globalPose1.position - joint->globalPose0.position);
			updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
				(joints + tid)->body1 ? &pose1 : 0,
				*(joints + tid));
			double newd = joint->normal.dot(joint->globalPose1.position - joint->globalPose0.position);
			double ome = (newd - curd) == 0 ? 0.0 : (-(1.0 + joint->restitution)*joint->relVn) / (newd - curd);
			if (joint->body0)
				atomicAdd(omega + joint->bodyId0, ome / *(nConstraint+joint->bodyId0));
			if (joint->body1)
				atomicAdd(omega + joint->bodyId1, ome / *(nConstraint + joint->bodyId1));
		}
	}


	__global__ void visitAllJoints(PBDJoint<double>* joints, int nJoints, double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nJoints)
		{
			PBDJoint<double>* joint = joints + tid;

			BodyPose<double> pose0;
			if ((joints + tid)->body0)
				pose0 = (joints + tid)->body0->pose;
			BodyPose<double> pose1;
			if ((joints + tid)->body1)
				pose1 = (joints + tid)->body1->pose;
			updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
				(joints + tid)->body1 ? &pose1 : 0,
				*(joints + tid));
		}
	}

	void PBDSolver::GPUUpdate(Real dt)
	{
		int blockdim_body = m_blockdim;
		int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;
		int blockdim_joint = m_blockdim;
		int griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;

		//cudaError_t err1 = cudaGetLastError();

		if (m_broadPhaseDetection)
			m_broadPhaseDetection(this, dt);

		

		double subdt = dt / m_numSubstep;
		for (int i = 0; i < m_numSubstep; ++i)
		{
			

			if(m_nBodies>0)
			{
				// integrate body.
				PBD_integrateAndPrepareBody << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
					m_GPUPosChange.begin(), m_GPURotChange.begin(),
					m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
					m_nBodies, subdt);
				//cudaDeviceSynchronize();
				//err1 = cudaGetLastError();
				cuSynchronize();
			}

			// contact detection.
			synFromBodiedToRigid();
			this->doCustomUpdate(dt);

			if (m_narrowPhaseDetection)
			{
				m_narrowPhaseDetection(this, dt);
			}

			blockdim_body = m_blockdim;
			griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;
			blockdim_joint = m_blockdim;
			griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;



			//if (m_nBodies > 0)
			//{
			//	cudaMemset(m_GPUOmega.begin(), 0, sizeof(double) * m_nBodies);

			//}

			//if (m_nJoints > 0)
			//{
			//	PBD_solveCollisionPose << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
			//		m_GPUPosChange.begin(), m_GPURotChange.begin(), m_GPUConstraintCount.begin(), m_nBodies,
			//		subdt);
			//	cudaDeviceSynchronize();
			//	err1 = cudaGetLastError();
			//	cuSynchronize();
			//}

			//if (m_nJoints > 0)
			//{
			//	PBD_omegaCorrection << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
			//		m_GPUPosChange.begin(), m_GPURotChange.begin(), m_GPUOmega.begin(), m_GPUConstraintCount.begin(),
			//		m_nBodies, subdt);
			//	cudaDeviceSynchronize();
			//	err1 = cudaGetLastError();
			//	cuSynchronize();
			//}

			//if (m_nBodies > 0 && m_nJoints>0)
			//{
			//	PBD_updatePosChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
			//		m_GPUPosChange.begin(), m_GPURotChange.begin(), m_GPUConstraintCount.begin(),
			//		m_nBodies, subdt, m_GPUOmega.begin());
			//	cudaDeviceSynchronize();
			//	err1 = cudaGetLastError();
			//	cuSynchronize();
			//}

			// solve position constraints.
			if (m_nJoints > 0)
			{
				PBD_updateConstraintNumber << <griddim_joint, blockdim_joint >> > (
					m_GPUJoints.begin(), m_nJoints
					);

				PBD_solveJoints << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
					m_GPUPosChange.begin(), m_GPURotChange.begin(), m_GPUConstraintCount.begin(), m_nBodies,
					subdt);
				//cudaDeviceSynchronize();
				//err1 = cudaGetLastError();
				cuSynchronize();
			}

			// update velocity and position changes.
			if (m_nBodies > 0)
			{
				PBD_updatePosChangeAndVelocity << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(), m_nBodies,
					m_GPUPosChange.begin(), m_GPURotChange.begin(),/* m_GPUConstraintCount.begin(),*/
					subdt, 1.0);
				cuSynchronize();
			}

			//if (m_nJoints > 0)
			//{
			//	visitAllJoints << < griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints, subdt);
			//}

			//Array<PBDBodyInfo<double>, DeviceType::CPU> cpuBodys;
			//cpuBodys.resize(m_GPUBodys.size());
			//Function1Pt::copy(cpuBodys, m_GPUBodys);

			//Array<PBDJoint<double>, DeviceType::CPU> cpuJoints;
			//cpuJoints.resize(m_GPUJoints.size());
			//Function1Pt::copy(cpuJoints, m_GPUJoints);

			//HostArray<Vector3d> hostdx;
			//hostdx.resize(m_GPUPosChange.size());
			//Function1Pt::copy(hostdx, m_GPUPosChange);

			

			// solve velocity constraints.
			m_GPULinvChange.reset();
			m_GPUAngvChange.reset();

			//PBD_solveVelocity << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
			//	m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
			//	m_nBodies, subdt);
			//cudaDeviceSynchronize();
			//err1 = cudaGetLastError();
			//cuSynchronize();

			//PBD_updateVelocityChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
			//	m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
			//	m_nBodies, subdt, 1.0);
			//cuSynchronize();


			//Array<Vector3d, DeviceType::CPU> cpuDLinv;
			//cpuDLinv.resize(m_GPUAngvChange.size());
			//Function1Pt::copy(cpuDLinv, m_GPUAngvChange);

			//Function1Pt::copy(cpuBodys, m_GPUBodys);

			//cpuDLinv.release();
			//cpuBodys.release();
			//cpuJoints.release();
			//hostdx.release();

			synFromBodiedToRigid();
		}



		//cudaDeviceSynchronize();
		//err1 = cudaGetLastError();
		//cuSynchronize();

		//Function1Pt::copy(m_CPUBodys, m_GPUBodys);
		

	}


	void PBDSolver::integrateBodyForce(Real dt)
	{
		if (m_nBodies > 0)
		{
			cuExecute(m_nBodies, PBD_integrateForce,
				m_GPUBodys.begin(), m_nBodies, dt
			);
		}
	}

	void PBDSolver::integrateBodyVelocity(Real dt)
	{
		if (m_nBodies > 0)
		{
			cuExecute(m_nBodies, PBD_integrateVelocity,
				m_GPUBodys.begin(),
				m_GPUPosChange.begin(), m_GPURotChange.begin(),
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
				m_nBodies, dt
			);

		}

	}

	void PBDSolver::solveSubStepGPU(Real dt)
	{
		int blockdim_body = m_blockdim;
		int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;
		int blockdim_joint =  m_blockdim;
		int griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;

		// contact detection.
		m_timer.start();
		synFromBodiedToRigid();
		this->doCustomUpdate(dt);

		if (m_narrowPhaseDetection)
		{
			m_narrowPhaseDetection(this, dt);
		}
		m_timer.stop();
		//printf("Narrow phase detect: %lf s\n", m_timer.getElapsedTime());

		blockdim_body = m_blockdim;
		griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;
		blockdim_joint = m_blockdim;
		griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;


		// solve position constraints.
		if (m_nJoints > 0)
		{
			//cudaError_t err = cudaGetLastError();

			PBD_updateConstraintNumber << <griddim_joint, blockdim_joint >> > (
				m_GPUJoints.begin(), m_nJoints
				);

			//err = cudaGetLastError();


			//HostArray<PBDJoint<double>> hostjoint;
			//hostjoint.resize(m_GPUJoints.size());
			//Function1Pt::copy(hostjoint, m_GPUJoints);


			//hostjoint.release();

			PBD_solveJoints << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
				m_GPUPosChange.begin(), m_GPURotChange.begin(), m_GPUConstraintCount.begin(), m_nBodies,
				dt);

			//cudaDeviceSynchronize();
			//err = cudaGetLastError();

			cuSynchronize();
		}

		// update velocity and position changes.
		if (m_nBodies > 0)
		{
			PBD_updatePosChangeAndVelocity << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(), m_nBodies,
				m_GPUPosChange.begin(), m_GPURotChange.begin(), /*m_GPUConstraintCount.begin(),*/
				dt, 1.0);
			cuSynchronize();
		}

		// solve velocity constraints.
		if (m_nJoints > 0)
		{
			m_GPULinvChange.reset();
			m_GPUAngvChange.reset();

			PBD_solveVelocity << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
				m_nBodies, dt);
			cudaDeviceSynchronize();
			cuSynchronize();

			PBD_updateVelocityChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
				m_nBodies, dt, 1.0);
			cuSynchronize();
		}
	}

	//void PBDSolver::integrateMotion(Real dt)
	//{
	//	if (m_nBodies <= 0)
	//		return;

	//	if (m_useGPU)
	//	{
	//		cuExecute(m_nBodies, PBD_integrateAndPrepareBody,
	//			m_GPUBodys.begin(),
	//			m_GPUPosChange.begin(), m_GPURotChange.begin(),
	//			m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
	//			m_nBodies, dt
	//		);
	//	}
	//	else
	//	{
	//		for (int i = 0; i < m_nBodies; ++i)
	//		{
	//			auto& body = m_CPUBodys[i];
	//			body.integrate(dt);
	//		}
	//	}
	//}

	//void PBDSolver::solvePosition(Real dt)
	//{
	//	if (m_nJoints <= 0)
	//		return;

	//	if (m_useGPU)
	//	{
	//		cuExecute(m_nJoints, PBD_updateConstraintNumber,
	//			m_GPUJoints.begin(), m_nJoints
	//		);

	//		cuExecute(m_nJoints, PBD_solveJoints,
	//			m_GPUJoints.begin(), m_nJoints,
	//			m_GPUPosChange.begin(), m_GPURotChange.begin(), m_GPUConstraintCount.begin(), m_nBodies,
	//			dt
	//		);
	//	}
	//	else
	//	{
	//		for (int i = 0; i < m_nJoints; ++i)
	//		{
	//			auto& joint = m_CPUJoints[i];
	//			joint.solvePose(dt);
	//		}
	//	}

	//}

	void PBDSolver::forwardSubStepGPU(Real dt)
	{
		// Call custom update function.
		this->doCustomUpdate(dt);

		//printf("\n");

		// Update body data.
		if (m_bodyDataDirty)
		{
			_onRigidDataDirty();
			m_bodyDataDirty = false;
		}


		// Integrate position and velocity.
		if (m_nBodies > 0)
		{
			int blockdim_body = m_blockdim;
			int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;

			// integrate body.
			PBD_integrateAndPrepareBody << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
				m_GPUPosChange.begin(), m_GPURotChange.begin(),
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
				m_nBodies, dt);
			cuSynchronize();
		}

		synFromBodiedToRigid();

		CTimer curtimer;
		curtimer.start();
		// Do contact detection.
		if (m_narrowPhaseDetection)
		{
			m_narrowPhaseDetection(this, dt);
		}
		curtimer.stop();
		m_detectionTime += curtimer.getElapsedTime();
		//printf("Detection time: %lf\n", curtimer.getElapsedTime());

		// Solve none-contact constraints.
		if (/*m_nPermanentJoints*/m_nJoints > 0)
		{
			int blockdim_joint = 256;//m_blockdim;
			int griddim_joint = (m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;
			int blockdim_body = m_blockdim;
			int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;

			griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;
			// Update permanent constraint number info.
			PBD_updateConstraintNumber << <griddim_joint, blockdim_joint >> > (
				m_GPUJoints.begin(), /*m_nPermanentJoints */m_nJoints
				);
			cuSynchronize();

			for (int iteri = 0; iteri < m_numContactSolveIter; ++iteri)
			{
				cuSafeCall(cudaMemset(m_GPUPosChange.begin(), 0, sizeof(Vector3d) * m_GPUPosChange.size()));
				cuSafeCall(cudaMemset(m_GPURotChange.begin(), 0, sizeof(Quaterniond) * m_GPURotChange.size()));

				//if (iteri >= m_numContactSolveIter - 180)
				if (m_nPermanentJoints > 0)
				{
					griddim_joint = (m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;
					// Solve permanent constraint.
					PBD_solveConstraintJoints << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nPermanentJoints,
						m_GPUPosChange.begin(), m_GPURotChange.begin(), dt
						);
					cuSynchronize();
				}

				if (/*iteri >= m_numContactSolveIter -1&&*/ m_nJoints > m_nPermanentJoints)
				{
					//synFromBodiedToRigid();


					griddim_joint = (m_nJoints - m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;
					// Solve contact joints.
					PBD_solveContactJoints << <griddim_joint, blockdim_joint >> > (
						m_GPUJoints.begin() + m_nPermanentJoints, m_nJoints - m_nPermanentJoints,
						m_GPUPosChange.begin(), m_GPURotChange.begin(), dt
						);
					cuSynchronize();
				}

				// Update position change.
				// Constraint number will be reset.
				PBD_updatePosChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
					m_GPUPosChange.begin(), m_GPURotChange.begin(), /*m_GPUConstraintCount.begin(),*/
					m_nBodies,
					dt, 1.0,
					iteri == m_numContactSolveIter - 1
					);
				cuSynchronize();

				// Solve velocity constraints.
				if (m_nJoints > 0 && m_solveVelocity)
				{
					griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;

					m_GPULinvChange.reset();
					m_GPUAngvChange.reset();

					PBD_solveVelocity << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
						m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
						m_nBodies, dt);
					cuSynchronize();

					PBD_updateVelocityChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
						m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
						m_nBodies, dt, 1.0);
					cuSynchronize();
				}
			}
		}

		// Solve velocity constraints.
		//if (m_nJoints > 0 && m_solveVelocity)
		if(false)
		{
			int blockdim_joint = 256;//m_blockdim;
			int griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;
			int blockdim_body = m_blockdim;
			int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;

			m_GPULinvChange.reset();
			m_GPUAngvChange.reset();

			PBD_solveVelocity << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
				m_nBodies, dt);
			cuSynchronize();

			PBD_updateVelocityChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
				m_nBodies, dt, 1.0);
			cuSynchronize();
		}

		synFromBodiedToRigid();
	}

	void PBDSolver::forwardSubStepGPU2(Real dt)
	{
		// Call custom update function.
		this->doCustomUpdate(dt);

		//printf("\n");

		// Update body data.
		if (m_bodyDataDirty)
		{
			_onRigidDataDirty();
			m_bodyDataDirty = false;
		}


		// Integrate position and velocity.
		if (m_nBodies > 0)
		{
			int blockdim_body = m_blockdim;
			int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;

			// integrate body.
			PBD_integrateAndPrepareBody << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
				m_GPUPosChange.begin(), m_GPURotChange.begin(),
				m_GPULinvChange.begin(), m_GPUAngvChange.begin(), m_GPUConstraintCount.begin(),
				m_nBodies, dt);
			cuSynchronize();
		}

		synFromBodiedToRigid();

		// Do contact detection.
		if (m_narrowPhaseDetection)
		{
			m_narrowPhaseDetection(this, dt);
		}

		// Solve none-contact constraints.
		if (/*m_nPermanentJoints*/m_nJoints > 0)
		{
			int blockdim_joint = 256;//m_blockdim;
			int griddim_joint = (m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;
			int blockdim_body = m_blockdim;
			int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;

			griddim_joint = (m_nJoints + blockdim_joint - 1) / blockdim_joint;
			// Update permanent constraint number info.
			PBD_updateConstraintNumber << <griddim_joint, blockdim_joint >> > (
				m_GPUJoints.begin(), /*m_nPermanentJoints */m_nJoints
				);
			cuSynchronize();

			for (int iteri = 0; iteri < m_numContactSolveIter; ++iteri)
			{
				cuSafeCall(cudaMemset(m_GPUPosChange.begin(), 0, sizeof(Vector3d) * m_GPUPosChange.size()));
				cuSafeCall(cudaMemset(m_GPURotChange.begin(), 0, sizeof(Quaterniond) * m_GPURotChange.size()));

				//if (iteri >= m_numContactSolveIter - 180)
				if(m_nPermanentJoints>0)
				{
					griddim_joint = (m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;
					// Solve permanent constraint.
					PBD_solveConstraintJoints << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nPermanentJoints,
						m_GPUPosChange.begin(), m_GPURotChange.begin(), dt
						);
					//cuSynchronize();
				}

				//if(false)
				if (/*iteri >= m_numContactSolveIter -1&&*/ m_nJoints > m_nPermanentJoints)
				{
					//synFromBodiedToRigid();


					griddim_joint = (m_nJoints - m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;
					// Solve contact joints.
					PBD_solveContactJoints << <griddim_joint, blockdim_joint >> > (
						m_GPUJoints.begin() + m_nPermanentJoints, m_nJoints - m_nPermanentJoints,
						m_GPUPosChange.begin(), m_GPURotChange.begin(), dt
						);
					//cuSynchronize();
				}

				// Update position change.
				// Constraint number will be reset.
				PBD_updatePosChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
					m_GPUPosChange.begin(), m_GPURotChange.begin(), /*m_GPUConstraintCount.begin(),*/
					m_nBodies,
					dt, 1.0, 
					iteri == m_numContactSolveIter - 1
					);
				cuSynchronize();
			}
		}

		//synFromBodiedToRigid();

		//// Do contact detection.
		//if (m_narrowPhaseDetection)
		//{
		//	m_narrowPhaseDetection(this, dt);
		//}


		// Solve contact constraints.
		//if (m_nJoints - m_nPermanentJoints > 0)
		//if(false)
		//{
		//	int blockdim_joint = 256;//m_blockdim;
		//	int griddim_joint = (m_nJoints - m_nPermanentJoints + blockdim_joint - 1) / blockdim_joint;

		//	int blockdim_body = m_blockdim;
		//	int griddim_body = (m_nBodies + blockdim_body - 1) / blockdim_body;

		//	
		//	// Update contact constraint number info.
		//	PBD_updateConstraintNumber << <griddim_joint, blockdim_joint >> > (
		//		m_GPUJoints.begin() + m_nPermanentJoints, m_nJoints - m_nPermanentJoints
		//		);
		//	cuSynchronize();

		//	for (int iteri = 0; iteri < m_numContactSolveIter; ++iteri)
		//	{
		//		cuSafeCall(cudaMemset(m_GPUPosChange.begin(),0, sizeof(Vector3d) * m_GPUPosChange.size()));
		//		cuSafeCall(cudaMemset(m_GPURotChange.begin(), 0, sizeof(Quaterniond) * m_GPURotChange.size()));

		//		// Solve contact joints.
		//		PBD_solveContactJoints << <griddim_joint, blockdim_joint >> > (
		//			m_GPUJoints.begin() + m_nPermanentJoints, m_nJoints - m_nPermanentJoints,
		//			m_GPUPosChange.begin(), m_GPURotChange.begin(), dt
		//			);
		//		cuSynchronize();

		//		// Update position change.
		//		PBD_updatePosChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(), 
		//			m_GPUPosChange.begin(), m_GPURotChange.begin(), /*m_GPUConstraintCount.begin(),*/
		//			m_nBodies,
		//			dt, 1.0, 
		//			iteri == m_numContactSolveIter - 1
		//			);
		//		cuSynchronize();
		//	}
		//}
		
		// Update velocity .
		if (m_nBodies)
		{
			cuExecute(m_nBodies, PBD_updateVelocity,
				m_GPUBodys.begin(), m_nBodies, dt);
		}


		//// solve velocity constraints.
		//if (m_nJoints > 0 && m_solveVelocity)
		//{
		//	for (int velsolvei = 0; velsolvei < 1; ++velsolvei)
		//	{
		//		m_GPULinvChange.reset();
		//		m_GPUAngvChange.reset();

		//		PBD_solveVelocity << <griddim_joint, blockdim_joint >> > (m_GPUJoints.begin(), m_nJoints,
		//			m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
		//			m_nBodies, dt);
		//		//cudaDeviceSynchronize();
		//		//err1 = cudaGetLastError();
		//		cuSynchronize();


		//		PBD_updateVelocityChange << <griddim_body, blockdim_body >> > (m_GPUBodys.begin(),
		//			m_GPULinvChange.begin(), m_GPUAngvChange.begin(), /*m_GPUConstraintCount.begin(),*/
		//			m_nBodies, dt, 1.0);
		//		cuSynchronize();
		//	}
		//}

		synFromBodiedToRigid();


		//// debug
		//for (auto prigid : m_rigids)
		//{
		//	Vector3f pos = prigid->getGlobalR();
		//	Quaternionf rot = prigid->getGlobalQ();

		//	printf("Rigid pose: %lf %lf %lf,  %lf %lf %lf %lf\n", pos[0], pos[1], pos[2], rot[0], rot[1], rot[2], rot[3]);
		//}
		//printf("\n");
	}



	void PBDSolver::forwardSubStepCPU(Real dt)
	{
		// Integrate position and velocity.
		for (int i = 0; i < m_nBodies; ++i)
		{
			auto& body = m_CPUBodys[i];
			body.integrate(dt);
		}

		synFromBodiedToRigid();

		// Do custom update function.
		// Vehicle suspension spring can be update here.
		this->doCustomUpdate(dt);

		// Do contact detection.
		if (m_narrowPhaseDetection)
		{
			m_narrowPhaseDetection(this, dt);
		}

		// Solve position constraints.
		for (int i = 0; i < m_nJoints; ++i)
		{
			auto& joint = m_CPUJoints[i];
			joint.solvePose(dt);
		}

		// Update velocity according to position change.
		for (int i = 0; i < m_nBodies; ++i)
		{
			auto& body = m_CPUBodys[i];
			body.updateVelocity(dt);
		}

		// Solve velocity constraints.
		for (int i = 0; i < m_nJoints; ++i)
		{
			auto& joint = m_CPUJoints[i];
			joint.solveVelocity(dt);
		}

		synFromBodiedToRigid();
	}


	void PhysIKA::PBDSolver::forwardSubStepCPU2(Real dt)
	{
		// Do custom update function.
		// Vehicle suspension spring can be update here.
		this->doCustomUpdate(dt);

		// Update body data.
		if (m_bodyDataDirty)
		{
			_onRigidDataDirty();
			m_bodyDataDirty = false;
		}

		// Integrate position and velocity.
		for (int i = 0; i < m_nBodies; ++i)
		{
			auto& body = m_CPUBodys[i];
			body.integrate(dt);
		}
		
		synFromBodiedToRigid();

		// Do contact detection.
		if (m_narrowPhaseDetection)
		{
			m_narrowPhaseDetection(this, dt);
		}

		//// Solve none-contact constraints.
		//for (int i = 0; i < m_nPermanentJoints; ++i)
		//{
		//	auto& joint = m_CPUJoints[i];
		//	joint.solvePose(dt);
		//}

		

		// Solve all joints.
		// Do multi-iteration to reach a better solution.
		for (int iteri = 0; iteri < m_numContactSolveIter; ++iteri)
		{
			for (int i = 0; i < m_nJoints; ++i)
			{
				auto& joint = m_CPUJoints[i];
				joint.solvePose(dt);
			}
		}

		// Update velocity according to position change.
		for (int i = 0; i < m_nBodies; ++i)
		{
			auto& body = m_CPUBodys[i];
			body.updateVelocity(dt);

			body.globalDamping(dt);
		}

		//for (int i = 0; i < m_nJoints; ++i)
		//{
		//	auto& joint = m_CPUJoints[i];
		//	joint.solveVelocity(dt);
		//}

		synFromBodiedToRigid();
	}


	void PBDSolver::forwardSubStep(Real dt)
	{
		if (m_useGPU)
		{
			forwardSubStepGPU2(dt);
		}
		else
		{
			//if (m_bodyDataDirty)
			//{
			//	_onRigidDataDirty();
			//	m_bodyDataDirty = false;
			//}

			//if (m_broadPhaseDetection)
			//	m_broadPhaseDetection(this, dt*m_numSubstep);

			forwardSubStepCPU2(dt);

			//m_bodyDataDirty = false;
			//m_jontDataDirty = false;
		}
	}




	PBDParticleBodyContactSolver::PBDParticleBodyContactSolver()
	{
	}


	__global__ void PBD_PBC_solveParticleBodyContact(
		PBDJoint<double>* joints, int njoints,
		Vector3d* dx, Quaterniond* dq, double dt
	)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= njoints || !((joints + tid)->active)) return;

		PBDJoint<double>* joint = joints + tid;
		BodyPose<double> pose0;
		if ((joints + tid)->body0)
			pose0 = (joints + tid)->body0->pose;
		
		if (!((joints + tid)->body1))
			return;
		BodyPose<double>& pose1 = (joints + tid)->body1->pose;

		updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
			(joints + tid)->body1 ? &pose1 : 0,
			*(joints + tid));

		Vector3d pcorr = (joints + tid)->globalPose1.position - (joints + tid)->globalPose0.position;
		double dn = pcorr.dot(joint->normal);
		pcorr = (joint->normal) * dn;
		double lambda = 0.0;


		// limite normal position.
		if (dn < 0)
		{
			//printf("Pos corr: %d,   %lf %lf %lf\n ", joint->bodyId1, pcorr[0], pcorr[1], pcorr[2]);


			lambda = applyBodyPairCorrection(pose0, pose1,
				(joints + tid)->body0, (joints + tid)->body1,
				*(joints + tid),
				pcorr, dt, true);
			joint->normLambda = lambda / dt;
		}

		// limite tangent position
		updateGlobalPose((joints + tid)->body0 ? &pose0 : 0,
			(joints + tid)->body1 ? &pose1 : 0,
			*(joints + tid));
		pcorr = joint->globalPose1.position - joint->globalPose0.position;
		dn = pcorr.dot(joint->normal);
		pcorr -= (joint->normal) * dn;
		//if ((joints + tid)->beContact)
		{
			lambda *= joint->mu;

			pcorr = Vector3d(0.0, 0.0, 0.0);
			if (joint->body1)
			{
				pcorr = joint->localPose1.position;
				joint->body1->prevPose.transform(pcorr);
				pcorr = joint->globalPose1.position - pcorr;
			}
			Vector3d tmpv;
			if (joint->body0)
			{
				tmpv = joint->localPose0.position;
				joint->body0->prevPose.transform(tmpv);
				tmpv = joint->globalPose0.position - tmpv;
			}
			pcorr -= tmpv;
			pcorr = pcorr - joint->normal * pcorr.dot(joint->normal);
		}

		applyBodyPairCorrection(pose0, pose1,
			(joints + tid)->body0, (joints + tid)->body1,
			*(joints + tid),
			pcorr, dt, true, lambda);

		
		// relvn.
		joint->relVn = joint->getJointVelocity().dot(joint->normal);

		// Update postion change.
		if ((joints + tid)->body0)
		{

			PBDBodyInfo<double>* body = (joints + tid)->body0;
			Vector3d dpos = pose0.position - body->pose.position;
			Quaterniond drot = pose0.rotation - body->pose.rotation;
			int bodyid = (joints + tid)->bodyId0;
			positionAdd(*(dx + bodyid), dpos);
			rotationAdd(*(dq + bodyid), drot);
			//atomicAdd(&(body->nContacts), 1);
		}
	}

	__global__ void PBD_PBC_updateParticleVelocityChange(
		Vector3d* parVel,
		//Vector3d* parDx,
		PBDBodyInfo<double>* particle,
		int nparticle, double dt,
		double ome = 1.0
	)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= nparticle) return;

		PBDBodyInfo<double>* par = particle + tid;
		Vector3d dx = par->pose.position - par->prevPose.position;
		double alpha = ome;
		parVel[tid] += dx * (alpha / dt);
	}


	__global__ void PBD_PBC_updatePosChangeAndVelocityChange(PBDBodyInfo<double>* bodys, int nbodys,
		Vector3d* dx, Quaterniond* dq, /*int* nConstraint,*/
		double dt, double ome)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;

		if (tid < nbodys)
		{
			PBDBodyInfo<double>* body = (bodys + tid);
			if (body &&body->nContacts > 0)
			{
				double alpha = ome / (body->nContacts);// ome / *(nConstraint + tid);
				//alpha = alpha > 1.0 ? 1.0 : alpha;

				body->pose.position += (*(dx + tid)) * alpha;
				body->pose.rotation += (*(dq + tid)) * alpha;
				body->pose.rotation.normalize();
			}


			body->updateVelocityChange(dt);
			
		}
	}

	/**
	* @brief solve velocity constraints.
	*/
	__global__ void PBD_PBC_solveCollisionVelocity(PBDJoint<double>* joints, int nJoints,
		Vector3d* parVel,
		Vector3d* dlinv, Vector3d* dangv, int nbodys,
		double dt)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= nJoints || !((joints + tid)->active)) return;


		PBDJoint<double>* joint = joints + tid;
		Vector3d linv0(0, 0, 0);
		Vector3d angv0(0, 0, 0);
		if (joint->body0)
		{
			linv0 = joint->body0->linVelocity;
			angv0 = joint->body0->angVelocity;
		}
		Vector3d linv1(0, 0, 0);
		Vector3d angv1(0, 0, 0);
		if (joint->body1)
		{
			linv1 = parVel[joint->bodyId1];
		}

		// Handle restitution.
		if (joint->beContact && joint->relVn <= 0)
		{

			updateGlobalPose(joint->body0 ? &(joint->body0->pose) : 0,
				joint->body1 ? &(joint->body1->pose) : 0,
				*joint);

			Vector3d dv;
			if (joint->body0)
				dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
			if (joint->body1)
				dv += joint->body1->getVelocityAt(joint->globalPose1.position);
			double ev = -joint->restitution * joint->relVn;
			ev = dv.dot(joint->normal) - (ev > 2.0*dt*9.8 ? ev : 0);
			dv = joint->normal * ev;
			applyPairVelocityCorrection(linv0, angv0, linv1, angv1,
				joint->body0, joint->body1,
				*joint,
				dv, dt, true);


			// Frictional velocitty.
			dv[0] = 0.0;	dv[1] = 0.0;	dv[2] = 0.0;
			if (joint->body0)
				dv -= joint->body0->getVelocityAt(joint->globalPose0.position);
			if (joint->body1)
				dv += joint->body1->getVelocityAt(joint->globalPose1.position);

			Vector3d vt = dv - joint->normal * dv.dot(joint->normal);

			ev = vt.norm();
			ev = (ev == 0 || ev < (joint->mu * joint->normLambda / dt)) ? 1.0 : (joint->mu * joint->normLambda / dt / ev);
			dv = vt * ev;
			applyPairVelocityCorrection(
				linv0, angv0, linv1, angv1,
				joint->body0, joint->body1,
				*joint,
				dv, dt, true);
		}

		// Update postion change.
		if (joint->body0)
		{
			Vector3d dlinvi = linv0 - joint->body0->linVelocity;
			Vector3d dangvi = angv0 - joint->body0->angVelocity;
			int bodyid = joint->bodyId0;
			positionAdd(*(dlinv + bodyid), dlinvi);
			positionAdd(*(dangv + bodyid), dangvi);
		}
		if (joint->body1)
		{
			parVel[joint->bodyId1] = linv1;
		}


	}

	void PBDParticleBodyContactSolver::forwardSubStep(Real dt, bool updateVel)
	{
		if (!m_body || !m_particle || !m_joints)
			return;

		if (m_body->size() <= 0 || m_particle->size() <= 0 || m_joints->size() <= 0)
			return;

		m_bodyPosChange.resize(m_body->size());
		m_bodyPosChange.reset();
		m_bodyRotChange.resize(m_body->size());
		m_bodyRotChange.reset();

		// Solve contact.
		cuExecute(m_joints->size(), PBD_PBC_solveParticleBodyContact,
			m_joints->begin(), m_joints->size(),
			m_bodyPosChange.begin(),
			m_bodyRotChange.begin(),
			dt
		);


		HostDArray<Vector3d> hostdpos;
		hostdpos.resize(m_bodyPosChange.size());
		Function1Pt::copy(hostdpos, m_bodyPosChange);

		HostDArray<Quaterniond> hostdrot;
		hostdrot.resize(m_bodyRotChange.size());
		Function1Pt::copy(hostdrot, m_bodyRotChange);

		HostDArray<PBDJoint<double>> hostjoint;
		hostjoint.resize(m_joints->size());
		Function1Pt::copy(hostjoint, *m_joints);

		HostArray<PBDBodyInfo<double>> hostbody;
		hostbody.resize(m_body->size());
		Function1Pt::copy(hostbody, *m_body);

		HostArray<PBDBodyInfo<double>> hostparticle;
		hostparticle.resize(m_particle->size());
		Function1Pt::copy(hostparticle, *m_particle);

		hostdpos.release();
		hostdrot.release();
		hostjoint.release();
		hostbody.release();
		hostparticle.release();

		//printf("Has Contacts.\n");


		// Update rigid body position and rotation change.
		cuExecute(m_body->size(), PBD_updatePosChangeAndVelocity,
			m_body->begin(), m_body->size(),
			m_bodyPosChange.begin(),
			m_bodyRotChange.begin(),
			dt, m_omega, updateVel
		);

		//if (m_particleVel)
		if(false)
		{
			if (m_particleVel->size() != m_particle->size())
				m_particleVel->resize(m_particle->size());

			cuExecute(m_particleVel->size(), PBD_PBC_updateParticleVelocityChange,
				m_particleVel->begin(),
				m_particle->begin(),
				m_particleVel->size(),
				dt, m_omega
			);
		}

		this->solveCollisionVelocity(dt);

	}

	void PBDParticleBodyContactSolver::solveCollisionVelocity(Real dt)
	{
		m_bodyLinvChange.resize(m_body->size());
		m_bodyLinvChange.reset();
		m_bodyAngvChange.resize(m_body->size());
		m_bodyAngvChange.reset();
		cuExecute(m_joints->size(), PBD_PBC_solveCollisionVelocity,
			m_joints->begin(), m_joints->size(),
			m_particleVel->begin(),
			m_bodyLinvChange.begin(), m_bodyAngvChange.begin(), m_body->size(),
			dt
		);

		cuExecute(m_body->size(), PBD_updateVelocityChange,
			m_body->begin(),
			m_bodyLinvChange.begin(), m_bodyAngvChange.begin(),
			m_body->size(), dt, m_omega
		);
	}


	__global__ void PBD_PBC_updateParticleBodyInfo(
		PBDBodyInfo<double>* body,
		Vector3d* posArr,
		Vector3d* velArr,
		double* m0,
		int nbody,
		double mu 
	)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= nbody)return;

		PBDBodyInfo<double>* curbody = body + tid;
		curbody->prevPose.position = posArr[tid];
		curbody->prevPose.rotation = Quaterniond();
		curbody->pose.position = posArr[tid];
		curbody->pose.rotation = Quaterniond();
		curbody->linVelocity = velArr[tid];
		curbody->angVelocity = Vector3d();

		curbody->mu = mu;
		curbody->invMass = 1.0 / m0[tid];
		curbody->invInertia = Vector3d();
		curbody->nContacts = 0;

	}


	__global__ void PBD_PBC_buildContactJoint(
		PBDJoint<double>* joints,
		DeviceDArray<ContactInfo<double>> contacts,
		PBDBodyInfo<double>* rigidBody,
		PBDBodyInfo<double>* particleBody
	)
	{
		int tid = threadIdx.x + blockDim.x * blockIdx.x;
		if (tid >= contacts.size()) return;


		PBDJoint<double>* joint = joints + tid;
		joint->bodyId0 = contacts[tid].id0;
		joint->bodyId1 = contacts[tid].id1;
		joint->body0 = rigidBody + joint->bodyId0;
		joint->body1 = particleBody + joint->bodyId1;

		joint->globalPose0.position = contacts[tid].point0;
		joint->globalPose1.position = contacts[tid].point1;

		joint->mu = contacts[tid].mu;
		joint->normal = contacts[tid].normal;
		joint->beContact = true;
		joint->rotationXLimited = false;
		joint->rotationYLimited = false;
		joint->rotationZLimited = false;
		joint->positionLimited = true;
		joint->maxDistance = -1.0;
		joint->restitution = 0.0;

		joint->active = true;

		if (joint->bodyId0 >= 0)
			atomicAdd(&(joint->body0->nContacts), 1);
		if (joint->bodyId1 >= 0)
			atomicAdd(&(joint->body1->nContacts), 1);
		joint->updateLocalPoses();

	}
	bool PBDParticleBodyContactSolver::buildJoints()
	{
		if (!m_contacts || !m_joints || m_contacts->size()<=0)
			return false;

		if (m_joints->size() != m_contacts->size())
			m_joints->resize(m_contacts->size());

		//cudaError_t err = cudaGetLastError();

		//HostDArray< ContactInfo<double>> hostcontact;
		//hostcontact.resize(m_contacts->size());
		//Function1Pt::copy(hostcontact, *m_contacts);

		//hostcontact.release();

		cuExecute(m_contacts->size(), PBD_PBC_buildContactJoint,
			m_joints->begin(),
			*m_contacts,
			m_body->begin(),
			m_particle->begin()
		);
	}




	void PBDParticleBodyContactSolver::updateParticleBodyInfo( double mu)
	{
		if (!m_particlePos || !m_particle || m_particlePos->size()<=0) return;

		if (m_particle->size() != m_particlePos->size())
			m_particle->resize(m_particlePos->size());

		cuExecute(m_particle->size(), PBD_PBC_updateParticleBodyInfo,
			m_particle->begin(),
			m_particlePos->begin(),
			m_particleVel->begin(),
			m_particleMass->begin(),
			m_particlePos->size(),
			mu
		);
	}



	__global__ void PBD_DS_computeLambdas(
		DeviceDArray<double> lambdaArr,
		DeviceDArray<double> rhoArr,
		DeviceDArray<double> rhoTar,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		NeighborList<int> neighbors,
		SpikyKernel2D<double> kern,
		double smoothingLength,
		double dt,
		double eps = 0.1
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Vector3d pos_i = posArr[pId];

		double lamda_i = 0.0;
		Vector3d grad_ci;

		int nbSize = neighbors.getNeighborSize(pId);
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			double r = (pos_i - posArr[j]).norm();

			if (r > EPSILON)
			{
				//double massj = massArr[j];
				double massj = neighbors.getNeighborSize(j);
				massj = massArr[j];// massj > 0 ? massArr[j] / massj : massArr[j];

				Vector3d g = (pos_i - posArr[j]) * ((1.0 / r)* kern.Gradient(r, smoothingLength) * massj) ;
				grad_ci += g;
				lamda_i += g.dot(g) / massj;
			}
		}

		double massi = massArr[pId];// nbSize > 0 ? massArr[pId] / nbSize : massArr[pId];
		lamda_i += grad_ci.dot(grad_ci) / massi;

		double rho_2d_ = rhoTar[pId];

		lamda_i = -(rhoArr[pId] - rho_2d_) * rho_2d_ / (lamda_i + eps * rho_2d_ * rho_2d_);

		//// debug
		//if (lamda_i != 0)
		//	printf("%d :  lambda:  %lf\n", pId, lamda_i);

		lambdaArr[pId] = lamda_i > 0.0f ? 0.0f : lamda_i;
	}


	__global__ void PBD_DS_computeDisplacement(
		DeviceDArray<Vector3d> dPos,
		DeviceDArray<double> lambdas,
		DeviceDArray<double> rhoTar,
		DeviceDArray<Vector3d> posArr,
		DeviceDArray<double> massArr,
		NeighborList<int> neighbors,
		SpikyKernel2D<double> kern,
		double smoothingLength,
		double dt)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= posArr.size()) return;

		Vector3d pos_i = posArr[pId];
		double lamda_i = lambdas[pId];
		double massi = massArr[pId];
		double rho_i_ = rhoTar[pId];
		Vector3d dpi;

		int nbSize = neighbors.getNeighborSize(pId);
		//massi = nbSize > 0 ? massi / nbSize : massi;
		for (int ne = 0; ne < nbSize; ne++)
		{
			int j = neighbors.getElement(pId, ne);
			double r = (pos_i - posArr[j]).norm();
			if (r > EPSILON)
			{
				double massj = neighbors.getNeighborSize(j);
				massj = massArr[j];// massj > 0 ? massArr[j] / massj : massArr[j];

				Vector3d dp_ij = /*10.0f**/(pos_i - posArr[j])*
					((lamda_i*massj/rho_i_ + lambdas[j]* massi / rhoTar[j])
						*kern.Gradient(r, smoothingLength)* (1.0 / r));
				dpi += dp_ij;
			}
		}
		dPos[pId] = dpi / massArr[pId] ;/*/(nbSize>0?nbSize:1)*/;
		if (nbSize <= 0)
			dPos[pId] = Vector3d();
	}


	__global__ void PBD_DS_updateParticleVelocityChange(
		Vector3d* parVel,
		//Vector3d* parDx,
		Vector3d* dPos,
		int nparticle, double dt
	)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= nparticle) return;

		parVel[tid] += dPos[tid] / dt;
		parVel[tid][1] = 0.0;
	}

	__global__ void PBD_DS_computeTargetRho(
		DeviceDArray<double> rhoTar,
		DeviceDArray<double> rhoArr,
		DeviceDArray<Vector3d> parVel,
		double rho0, double minh, double dt
	)
	{
		int tid = threadIdx.x + blockIdx.x * blockDim.x;
		if (tid >= rhoArr.size()) return;

		double rho_ = rhoArr[tid] + parVel[tid][1] * dt * rho0;
		double minrho = minh * rho0;
		if (rho_ < minrho)
			rho_ = minrho;
		rhoTar[tid] = rho_;
	}

	void PBDDensitySolver2D::forwardOneSubStep(Real dt)
	{
		if (!m_particlePos || !m_particleVel ||
			!m_particleRho2D || m_neighbor.isEmpty())
			return;

		int nsize = m_particlePos->size();
		// compute target rho.
		if (m_targetRho.size() != nsize)
			m_targetRho.resize(nsize);
		cuExecute(nsize, PBD_DS_computeTargetRho,
			m_targetRho,
			*m_particleRho2D,
			*m_particleVel,
			m_rho0, minh, dt
		);

		for (int i = 0; i < 1; ++i)
		{
			// Compute lambda.
			if (m_lambda.size() != nsize)
				m_lambda.resize(nsize);
			cuExecute(nsize, PBD_DS_computeLambdas,
				m_lambda,
				*m_particleRho2D,
				m_targetRho,
				*m_particlePos,
				*m_particleMass,
				m_neighbor.getValue(),
				m_kernel,
				m_smoothLength, dt, 0.01
			);

			// Compute dx.
			if (m_particleDpos.size() != nsize)
				m_particleDpos.resize(nsize);
			cuExecute(nsize, PBD_DS_computeDisplacement,
				m_particleDpos,
				m_lambda,
				m_targetRho,
				*m_particlePos,
				*m_particleMass,
				m_neighbor.getValue(),
				m_kernel,
				m_smoothLength,
				dt
			);


			//// debug
			//HostDArray<Vector3d> dpos;
			//dpos.resize(m_particleDpos.size());
			//Function1Pt::copy(dpos, m_particleDpos);

			//HostDArray<Vector3d> hostpos;
			//hostpos.resize(m_particlePos->size());
			//Function1Pt::copy(hostpos, *m_particlePos);

			//HostDArray<double> hostTarrho;
			//hostTarrho.resize(m_targetRho.size());
			//Function1Pt::copy(hostTarrho, m_targetRho);

			//HostDArray<Vector3d> hostVel;
			//hostVel.resize(m_particleVel->size());
			//Function1Pt::copy(hostVel, *m_particleVel);
			//
			//dpos.release();
			//hostTarrho.release();
			//hostpos.release();


			// Compute velocity change.
			cuExecute(nsize, PBD_DS_updateParticleVelocityChange,
				m_particleVel->begin(),
				m_particleDpos.begin(),
				nsize, dt
			);
		}

		//Function1Pt::copy(hostVel, *m_particleVel);
		//hostVel.release();
	}

}