#pragma once

#ifndef PHYSIKA_PBDBODYINFO_H
#define PHYSIKA_PBDBODYINFO_H

#include "Core/Platform.h"
#include "Core/Vector/vector_3d.h"
#include "Core/DataTypes.h"
#include "Core/Quaternion/quaternion.h"
#include "Dynamics/RigidBody/RigidBody2.h"

namespace PhysIKA {

template <typename Real>
class BodyPose
{
public:
    COMM_FUNC BodyPose()
        : position(0, 0, 0), rotation(0, 0, 0, 1)
    {
    }
    COMM_FUNC BodyPose& operator=(const BodyPose& p)
    {
        position = p.position;
        rotation = p.rotation;
        return *this;
    }

    COMM_FUNC void rotate(Vector<Real, 3>& v) const
    {
        v = rotation.rotate(v);
    }

    COMM_FUNC void invRotate(Vector<Real, 3>& v) const
    {
        v = rotation.getConjugate().rotate(v);
    }

    COMM_FUNC void transform(Vector<Real, 3>& v) const
    {
        v = position + rotation.rotate(v);
    }

    COMM_FUNC void invTransform(Vector<Real, 3>& v) const
    {
        v -= position;
        v = rotation.getConjugate().rotate(v);
    }

    COMM_FUNC void tranformPose(BodyPose& pose) const
    {
        pose.rotation = this->rotation * pose.rotation;
        pose.position = position + rotation.rotate(pose.position);
    }

    COMM_FUNC void invTransformPose(BodyPose& pose) const
    {
        pose.rotation = this->rotation.getConjugate() * pose.rotation;
        pose.position = this->rotation.getConjugate().rotate(pose.position - position);
    }

public:
    Vector<Real, 3>  position;
    Quaternion<Real> rotation;
};

/**
    * @brief (Rigid) body info of PBD rigid body simulation.
    */
//template<typename TDataType>
template <typename Real>
class PBDBodyInfo
{
public:
    //typedef typename TDataType::Real Real;
    //typedef typename TDataType::Coord Coord;
    //typedef typename TDataType::Matrix Matrix;
    //typedef typename TDataType::Rigid Rigid;

    COMM_FUNC void globalDamping(Real dt)
    {
        Real linDamp = pow(1.0 - linDamping, dt);
        Real angDamp = pow(1.0 - angDamping, dt);

        linVelocity *= linDamp;
        angVelocity *= angDamp;
    }

    COMM_FUNC void integrate(Real dt)
    {
        prevPose = pose;
        linVelocity += externalForce * invMass * dt;

        //if(invInertia[0]==0 || invInertia[1] != 0 && invInertia[1] != 0)
        Vector<Real, 3> angv = angVelocity;
        pose.invRotate(angv);  // to local.
        Vector<Real, 3> omegaI(invInertia[0] == 0 ? 0 : angv[0] / invInertia[0],
                               invInertia[1] == 0 ? 0 : angv[1] / invInertia[1],
                               invInertia[2] == 0 ? 0 : angv[2] / invInertia[2]);
        omegaI               = angv.cross(omegaI);
        Vector<Real, 3> extT = externalTorque;
        pose.invRotate(extT);
        omegaI      = (extT - omegaI) * invInertia * dt;
        angVelocity = angv + omegaI;
        pose.rotate(angVelocity);

        pose.position += linVelocity * dt;
        this->integrateRotation(angVelocity, dt);

        //printf(" Velocity: %lf %lf %lf;  ExtF: %lf %lf %lf\n",
        //    linVelocity[0], linVelocity[1], linVelocity[2],
        //    externalForce[0], externalForce[1], externalForce[2]);
    }

    COMM_FUNC void integrateForce(Real dt)
    {
        linVelocity += externalForce * invMass * dt;

        //if(invInertia[0]==0 || invInertia[1] != 0 && invInertia[1] != 0)
        Vector<Real, 3> angv = angVelocity;
        pose.invRotate(angv);  // to local.
        Vector<Real, 3> omegaI(invInertia[0] == 0 ? 0 : angv[0] / invInertia[0],
                               invInertia[1] == 0 ? 0 : angv[1] / invInertia[1],
                               invInertia[2] == 0 ? 0 : angv[2] / invInertia[2]);
        omegaI               = angv.cross(omegaI);
        Vector<Real, 3> extT = externalTorque;
        pose.invRotate(extT);
        omegaI      = (extT - omegaI) * invInertia * dt;
        angVelocity = angv + omegaI;
        pose.rotate(angVelocity);
    }

    COMM_FUNC void integrateForce(const Vector<Real, 3>& extF, const Vector<Real, 3>& extT, Real dt)
    {
        linVelocity += extF * invMass * dt;

        //if(invInertia[0]==0 || invInertia[1] != 0 && invInertia[1] != 0)
        Vector<Real, 3> angv = angVelocity;
        pose.invRotate(angv);  // to local.
        Vector<Real, 3> omegaI(invInertia[0] == 0 ? 0 : angv[0] / invInertia[0],
                               invInertia[1] == 0 ? 0 : angv[1] / invInertia[1],
                               invInertia[2] == 0 ? 0 : angv[2] / invInertia[2]);
        omegaI                    = angv.cross(omegaI);
        Vector<Real, 3> extTLocal = extT;
        pose.invRotate(extTLocal);
        omegaI      = (extTLocal - omegaI) * invInertia * dt;
        angVelocity = angv + omegaI;
        pose.rotate(angVelocity);
    }

    COMM_FUNC void integrateForceToVelPos(const Vector<Real, 3>& extF, const Vector<Real, 3>& extT, Real dt)
    {

        Vector<Real, 3> dlinv = extF * invMass * dt;
        linVelocity += dlinv;
        pose.position += dlinv * dt;

        Vector<Real, 3> angv = angVelocity;
        pose.invRotate(angv);  // to local.
        Vector<Real, 3> dangv(invInertia[0] == 0 ? 0 : angv[0] / invInertia[0],
                              invInertia[1] == 0 ? 0 : angv[1] / invInertia[1],
                              invInertia[2] == 0 ? 0 : angv[2] / invInertia[2]);
        dangv                     = angv.cross(dangv);
        Vector<Real, 3> extTLocal = extT;
        pose.invRotate(extTLocal);
        dangv = (extTLocal /*- dangv*/) * invInertia * dt;
        pose.rotate(dangv);
        angVelocity += dangv;

        this->integrateRotation(dangv, dt);
    }

    COMM_FUNC void integrateVelocity(Real dt)
    {
        prevPose = pose;
        pose.position += linVelocity * dt;
        this->integrateRotation(angVelocity, dt);
    }

    COMM_FUNC void updateVelocity(Real dt)
    {
        // update linear velocity.
        linVelocity = (pose.position - prevPose.position) / dt;

        //printf(" Body Lin v:  %lf, %lf, %lf\n", linVelocity[0], linVelocity[1], linVelocity[2]);

        // update angular velocity.
        Quaternion<Real> relq = pose.rotation * prevPose.rotation.getConjugate();
        Real             fac  = 2.0 / dt;
        fac *= relq.w() < 0.0 ? -1.0 : 1.0;
        angVelocity = { relq.x() * fac, relq.y() * fac, relq.z() * fac };
    }

    COMM_FUNC void updateVelocityChange(Real dt)
    {
        // update linear velocity.
        linVelocity += (pose.position - prevPose.position) / dt;

        //printf(" Body Lin v:  %lf, %lf, %lf\n", linVelocity[0], linVelocity[1], linVelocity[2]);

        // update angular velocity.
        Quaternion<Real> relq = pose.rotation * prevPose.rotation.getConjugate();
        Real             fac  = 2.0 / dt;
        fac *= relq.w() < 0.0 ? -1.0 : 1.0;
        angVelocity += { relq.x() * fac, relq.y() * fac, relq.z() * fac };
    }

    /**
        * @brief Calculate PBD inverse mass value.
        * @param normal, direction of force or torque in WORLD frame.
        * @param pos, position of force in WORLD frame. 
        * @param bepostion, whether the position is valid. Set to FALSE if it is a rotational interaction.
        */
    COMM_FUNC Real getPBDInverseMass(const Vector<Real, 3>& normal, const Vector<Real, 3>& pos, bool beposition = true) const
    {
        Vector<Real, 3> n(normal);
        Real            w = 0;
        if (beposition)
        {
            n = (pos - pose.position).cross(normal);
            w = invMass;
        }

        pose.invRotate(n);
        w += n[0] * n[0] * invInertia[0] + n[1] * n[1] * invInertia[1] + n[2] * n[2] * invInertia[2];
        return w;
    }

    /**
        * @brief Integrate body rotation.
        * @param rot, angular velocity in WORLD frame.
        */
    COMM_FUNC void integrateRotation(const Vector<Real, 3>& rot, Real scale = 1.0)
    {
        Real phi = rot.norm();
        scale *= 0.5;
        Quaternion<Real> dq(rot[0] * scale, rot[1] * scale, rot[2] * scale, 0.0);
        pose.rotation = pose.rotation + dq * pose.rotation;
        pose.rotation.normalize();
    }

    /**
        * @brief Apply a postional or rotational correction to body.
        * @param corr, correction direction. The direction of displacement or rotation, in WORLD frame.
        * @param pos, GLOBAL position of interaction point, in WORLD frame.
        */
    COMM_FUNC void applyCorrection(const Vector<Real, 3>& corr, const Vector<Real, 3>& pos, bool beposition, bool velocityLevel = false)
    {
        Vector<Real, 3> dp(corr);

        // Position correction.
        if (beposition)
        {
            if (velocityLevel)
            {
                this->linVelocity += corr * invMass;
            }
            else
            {
                this->pose.position += corr * invMass;
            }
            dp = (pos - pose.position).cross(corr);
        }

        // rotation correction.
        this->pose.invRotate(dp);  // to local frame.
        dp *= invInertia;          // angular velocity in local frame.
        this->pose.rotate(dp);     // to world frame.
        if (velocityLevel)
        {
            this->angVelocity += dp;
        }
        else
        {
            this->integrateRotation(dp);
        }
    }

    COMM_FUNC Vector<Real, 3> getVelocityAt(const Vector<Real, 3>& p)
    {
        Vector<Real, 3> v = p - pose.position;
        v                 = angVelocity.cross(v);
        v += linVelocity;
        return v;
    }

public:
    BodyPose<Real> pose;      // body position and rotation in world frame.
    BodyPose<Real> prevPose;  // body position and rotation in last time step.
    //BodyPose<Real> preSubstePose;

    Vector<Real, 3> linVelocity;  // linear velocity(world frame).
    Vector<Real, 3> angVelocity;  // angular velocity(world frame).

    Real mu  = 1.0;
    Real rho = 1.0;

    Real linDamping = 0.;
    Real angDamping = 0.;

    Real            invMass;     // inverse mass.
    Vector<Real, 3> invInertia;  // inverse inertia(local frame).

    Vector<Real, 3> externalForce;   // exteranal center force. (Acceleration)
    Vector<Real, 3> externalTorque;  // exteranal center torque. (torque)

    int nContacts;
};

#ifdef PRECISION_FLOAT
//template class PBDBodyInfo<DataType3f>;
template class PBDBodyInfo<float>;
#else
//template class PBDBodyInfo<DataType3d>;
template class PBDBodyInfo<double>;
#endif

}  // namespace PhysIKA

#endif  // PHYSIKA_PBDBODYINFO_H