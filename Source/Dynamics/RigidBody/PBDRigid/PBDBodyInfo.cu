
#include "Dynamics/RigidBody/PBDRigid/PBDBodyInfo.h"

//template<typename Real>
//inline COMM_FUNC void PhysIKA::PBDBodyInfo<Real>::integrate(Real dt)
//{
//	prevPose = pose;
//	linVelocity += externalForce * dt;
//	angVelocity += externalTorque * dt;
//
//	pose.position += linVelocity * dt;
//	this->integrateRotation(angVelocity, dt);
//}
//
//template<typename Real>
//COMM_FUNC void PhysIKA::PBDBodyInfo<Real>::updateVelocity(Real dt)
//{
//	// update linear velocity.
//	linVelocity = (pose.position - prevPose.position) / dt;
//
//	// update angular velocity.
//	Quaternion<Real> relq = pose.rotation * prevPose.rotation.getConjugate();
//	Real fac = 2.0 / dt;
//	fac *= relq.w() < 0.0 ? -1.0 : 1.0;
//	angVelocity = { relq.x() * fac, relq.y() * fac, relq.z() * fac };
//}
//
//
//template<typename Real>
//COMM_FUNC Real PhysIKA::PBDBodyInfo<Real>::getPBDInverseMass(const Vector<Real, 3>& normal, const Vector<Real, 3>& pos, bool beposition)
//{
//	Vector<Real, 3> n(pos);
//	Real w = 0;
//	if (beposition)
//	{
//		n = n.cross(normal);
//		w = invMass;
//	}
//
//	pose.invRotate(n);
//	w += n[0] * n[0] * invInertia[0] + n[1] * n[1] * invInertia[1] + n[2] * n[2] * invInertia[2];
//	return w;
//}
//
//template<typename Real>
//COMM_FUNC void PhysIKA::PBDBodyInfo<Real>::integrateRotation(const Vector<Real, 3>& rot, Real scale)
//{
//	Real phi = rot.norm();
//	scale *= 0.5;
//	Quaternion<Real> dq(rot[0] * scale, rot[1] * scale, rot[2] * scale, 0.0);
//	pose.rotation = pose.rotation + dq * pose.rotation;
//	pose.rotation.normalize();
//}
//
//template<typename Real>
//COMM_FUNC void PhysIKA::PBDBodyInfo<Real>::applyCorrection(const Vector<Real, 3>& corr, const Vector<Real, 3>& pos, bool beposition)
//{
//	Vector<Real, 3> dp(corr);
//
//	// Position correction.
//	if (beposition)
//	{
//		this->pose.position += corr * invMass;
//		dp = (pos - pose.position).cross(corr);
//	}
//
//	// rotation correction.
//	this->pose.invRotate(dp);	// to local frame.
//	dp *= invInertia;			// angular velocity in local frame.
//	this->pose.rotate(dp);		// to world frame.
//	this->integrateRotation(dp);
//}
//
