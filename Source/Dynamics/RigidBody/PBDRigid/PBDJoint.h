#pragma once

#ifndef PHYSIKA_PBDJOINTINFO_H
#define PHYSIKA_PBDJOINTINFO_H

#include "Core/Platform.h"
#include "Core/Vector/vector_3d.h"
#include "Core/DataTypes.h"
#include "Core/Quaternion/quaternion.h"
#include "Dynamics/RigidBody/PBDRigid/PBDBodyInfo.h"

#include <math.h>
//#include "math_functions.h"
#include "math_constants.h"
#include <stdio.h>

namespace PhysIKA {

/**
    * @brief Joint info for PBD simulation.
    * @details A joint contain 2 kids of constraint: POSITION and ROTATION.
    *            Postion: constraint the distance between point pa(in body0) and point pb(inbody1) to be 0.
    *            Rotation: constraint the relative rotation between body0 and body1. If rotations were constraint fixed, 
    *                local axis X,Y,Z of body0 and body1 should align corespondently.
    */
//template<typename TDataType>
template <typename TReal>
class PBDJoint
{
public:
    //typedef typename TDataType::TReal TReal;
    //typedef typename TDataType::Coord Coord;
    //typedef typename TDataType::Matrix Matrix;
    //typedef typename TDataType::Rigid Rigid;

    //typedef typename PBDBodyInfo<TDataType> BodyInfoType;
    typedef typename PBDBodyInfo<TReal> BodyInfoType;

    COMM_FUNC PBDJoint()
        : body0(0), body1(0), compliance(0), rotationXLimited(true), minAngleX(0), maxAngleX(0), rotationYLimited(true), minAngleY(0), maxAngleY(0), rotationZLimited(true), minAngleZ(0), maxAngleZ(0), beContact(false), maxDistance(0)
    {
    }

    COMM_FUNC void updateGlobalPoses()
    {
        globalPose0 = localPose0;
        if (body0)
            body0->pose.tranformPose(globalPose0);
        globalPose1 = localPose1;
        if (body1)
            body1->pose.tranformPose(globalPose1);
    }

    COMM_FUNC void updateLocalPoses()
    {
        localPose0 = globalPose0;
        if (body0)
            body0->pose.invTransformPose(localPose0);
        localPose1 = globalPose1;
        if (body1)
            body1->pose.invTransformPose(localPose1);
    }

    COMM_FUNC Vector<TReal, 3> getJointVelocity()
    {
        Vector<TReal, 3> relv(0, 0, 0);
        if (body0)
            relv -= body0->getVelocityAt(globalPose0.position);
        if (body1)
            relv += body1->getVelocityAt(globalPose1.position);
        return relv;
    }

    /**
        * @brief Limit the angle between a and b within [minAngle, maxAngle]
        */
    COMM_FUNC void limitAngle(const Vector<TReal, 3>& n, const Vector<TReal, 3>& a, const Vector<TReal, 3>& b, TReal minAngle, TReal maxAngle, TReal dt, TReal maxCorr = CUDART_PI)
    {
        Vector<TReal, 3> axb = a.cross(b);
        TReal            phi = asin(axb.dot(n));
        if (a.dot(b) < 0)
            phi = CUDART_PI - phi;
        if (phi > CUDART_PI)
            phi -= 2.0 * CUDART_PI;
        if (phi < -CUDART_PI)
            phi += 2.0 * CUDART_PI;

        if (phi < minAngle || phi > maxAngle)
        {
            phi = minAngle > phi ? minAngle : phi;
            phi = maxAngle < phi ? maxAngle : phi;

            Quaternion<TReal> q(n, phi);
            Vector<TReal, 3>  omega = q.rotate(a);
            omega                   = omega.cross(b);

            phi = omega.norm();
            if (phi > maxCorr)
                omega *= maxCorr / phi;

            applyBodyPairCorrection(omega, dt, false);
        }
    }

    COMM_FUNC void updateContactActivation(TReal threashold)
    {
        if (beContact)
        {
            this->updateGlobalPoses();
            // debug
            //localPose0.position = globalPose1.position;
            //localPose0.position[1] = 0;;
            //this->updateGlobalPoses();

            Vector<TReal, 3> relp = globalPose1.position - globalPose0.position;
            if (relp.dot(this->normal) > 0)
            {
                this->active = false;
            }
            else
            {
                this->active = true;
            }
        }
    }

    COMM_FUNC void solvePose(TReal dt)
    {
        // update joint global pose.
        //this->updateGlobalPoses();

        if (!active)
            return;

        // Pose before solveing constraint.
        BodyPose<TReal> prepose0;
        BodyPose<TReal> prepose1;
        prepose0 = body0 ? body0->pose : prepose0;
        prepose1 = body1 ? body1->pose : prepose1;

        // *** SOLVE ROTATION ***
        // X rotation.
        if (rotationXLimited)
        {
            this->updateGlobalPoses();
            Vector<TReal, 3> a0 = _getQuatAxis0(globalPose0.rotation);
            Vector<TReal, 3> a1 = _getQuatAxis0(globalPose1.rotation);
            Vector<TReal, 3> n  = a0.cross(a1);
            if (n.norm() > 0 && a1.norm() > 0)
            {
                n.normalize();
                a1.normalize();
                //this->applyBodyPairCorrection(n, dt, false);

                this->limitAngle(n, a0, a1, minAngleX, maxAngleX, dt);
            }
        }

        // Y rotation. (or swing degree)
        if (rotationYLimited)
        {
            this->updateGlobalPoses();
            Vector<TReal, 3> b0 = _getQuatAxis1(globalPose0.rotation);
            Vector<TReal, 3> b1 = _getQuatAxis1(globalPose1.rotation);
            Vector<TReal, 3> n  = b0.cross(b1);
            if (n.norm() > 0 && b1.norm() > 0)
            {
                n.normalize();

                this->limitAngle(n, b0, b1, minAngleY, maxAngleY, dt);
            }
        }

        // Z rotation. (or twist degree)
        if (rotationZLimited)
        {
            this->updateGlobalPoses();
            Vector<TReal, 3> b0 = _getQuatAxis1(globalPose0.rotation);
            Vector<TReal, 3> b1 = _getQuatAxis1(globalPose1.rotation);
            Vector<TReal, 3> n  = (b0 + b1);
            n.normalize();

            Vector<TReal, 3> c0 = _getQuatAxis2(globalPose0.rotation);
            c0                  = (c0 - n * (c0.dot(n)));
            c0.normalize();
            Vector<TReal, 3> c1 = _getQuatAxis2(globalPose1.rotation);
            c1                  = (c1 - n * (c1.dot(n)));
            c1.normalize();

            this->limitAngle(n, c0, c1, minAngleZ, maxAngleZ, dt);
        }

        // ***** SOLVE POSITION.
        if (positionLimited)
        {
            this->updateGlobalPoses();

            Vector<TReal, 3> pcorr = globalPose1.position - globalPose0.position;
            Vector<TReal, 3> tmpv  = beContact ? this->normal : -(this->_getQuatAxis1(globalPose0.rotation));
            TReal            dn    = pcorr.dot(tmpv);
            if (this->beContact)
            {
                Vector<TReal, 3> prevpos1 = localPose1.position;
                if (body1)
                    body1->prevPose.transform(prevpos1);
                Vector<TReal, 3> prevpos0 = localPose0.position;
                if (body0)
                    body0->prevPose.transform(prevpos0);
                double prevpn = tmpv.dot(prevpos1 - prevpos0);

                double pn = -(this->restitution) * tmpv.dot(this->getJointVelocity()) * dt + prevpn;

                dn -= pn < 0 ? 0 : pn;
                //dn = dn < pn ? dn : pn;
                //dn = pn;
            }
            pcorr        = tmpv * dn;
            TReal lambda = 0.0;
            // limite normal position.
            if (dn < 0
                || (maxDistance >= 0 && dn > maxDistance))
            {
                if (maxDistance >= 0 && dn > maxDistance)
                    pcorr *= (dn - maxDistance) / dn;
                lambda = applyBodyPairCorrection(
                    pcorr, dt, true, -1.0, false, this->normLambda, dn > 0);
            }

            //debug
            //Vector<TReal, 3> corrn = pcorr;

            this->updateGlobalPoses();
            pcorr = globalPose1.position - globalPose0.position;

            //debug
            //double valdot = corrn.dot(pcorr);

            //tmpv = beContact ? this->normal : -(this->_getQuatAxis1(globalPose0.rotation));
            dn = pcorr.dot(tmpv);
            pcorr -= tmpv * dn;

            // limite tangent position
            if (beContact)
            {
                //this->updateGlobalPoses();
                this->normLambda += lambda;

                lambda = abs(/*this->normLambda*/ lambda * mu);  // -this->tangLambda;
                pcorr  = Vector<TReal, 3>(0, 0, 0);
                if (body1)
                {
                    pcorr = localPose1.position;
                    body1->prevPose.transform(pcorr);
                    pcorr = globalPose1.position - pcorr;
                }
                tmpv = Vector<TReal, 3>(0, 0, 0);
                if (body0)
                {
                    tmpv = localPose0.position;
                    body0->prevPose.transform(tmpv);
                    tmpv = globalPose0.position - tmpv;
                }
                pcorr = pcorr - tmpv;
                pcorr = pcorr - this->normal * pcorr.dot(this->normal);
            }
            else
            {
                lambda = -1.0;
            }
            //applyBodyPairCorrection(
            //    pcorr, dt, true, lambda);
            //lambda = -1.0;
            lambda = applyBodyPairCorrection(
                pcorr, dt, true, lambda, false, 0);

            // update normal velocity
            if (beContact)
            {
                this->tangLambda0 += lambda;

                //pcorrn[0] = 0.0;    pcorrn[1] = 0.0;    pcorrn[2] = 0.0;
                Vector<TReal, 3> relv = getJointVelocity();
                this->relVn           = relv.dot(normal);
            }
        }
    }

    COMM_FUNC void solveVelocity(TReal dt)
    {
        if (!active)
            return;

        this->updateGlobalPoses();

        // Damp rotation velocity.
        Vector<TReal, 3> dv(0, 0, 0);
        if (body0)
            dv -= this->body0->angVelocity;
        if (this->body1)
            dv += this->body1->angVelocity;
        dv *= this->angDamping * dt > 1.0 ? 1.0 : this->angDamping * dt;
        applyBodyPairCorrection(
            dv, dt, false, -1.0, true);

        // Damp linear velocity.
        dv[0] = 0.0;
        dv[1] = 0.0;
        dv[2] = 0.0;
        if (this->body0)
            dv -= this->body0->getVelocityAt(this->globalPose0.position);
        if (this->body1)
            dv += this->body1->getVelocityAt(this->globalPose1.position);
        dv *= this->linDamping * dt > 1.0 ? 1.0 : this->linDamping * dt;
        applyBodyPairCorrection(
            dv, dt, true, -1.0, true);

        // Handle restitution.
        if (this->beContact && this->relVn <= 0)
        {
            // Normal velocity.
            dv[0] = 0.0;
            dv[1] = 0.0;
            dv[2] = 0.0;
            if (this->body0)
                dv -= this->body0->getVelocityAt(this->globalPose0.position);
            if (this->body1)
                dv += this->body1->getVelocityAt(this->globalPose1.position);
            //TReal vn = dv.dot(this->normal);
            TReal ev = -this->restitution * this->relVn;
            ev       = dv.dot(this->normal) - (ev > 2.0 * dt * 9.8 ? ev : 0);
            dv       = this->normal * ev;
            applyBodyPairCorrection(
                dv, dt, true, -1.0, true);

            // Frictional velocitty.
            dv[0] = 0.0;
            dv[1] = 0.0;
            dv[2] = 0.0;
            if (this->body0)
                dv -= this->body0->getVelocityAt(this->globalPose0.position);
            if (this->body1)
                dv += this->body1->getVelocityAt(this->globalPose1.position);

            Vector<TReal, 3> vt = dv - this->normal * dv.dot(this->normal);

            ev = vt.norm();
            ev = (ev == 0 || 1.0 < abs(mu * normLambda / dt / ev)) ? 1.0 : abs(mu * normLambda / dt / ev);
            dv = vt * ev;
            //applyBodyPairCorrection(
            //    vt, dt, true, -1.0, true);
        }
    }

    COMM_FUNC void testGhostInitPose(Real dt)
    {
        if (active && beContact)
        {
            Vector<TReal, 3> jv  = getJointVelocity();
            TReal            jvn = jv.dot(this->normal);
            if (jvn >= 0)
                return;

            this->updateGlobalPoses();
            Vector<TReal, 3> relp  = globalPose1.position - globalPose0.position;
            TReal            relpn = relp.dot(this->normal);
            Vector<TReal, 3> dx    = this->normal * (relpn - jvn * restitution * dt);
            //BodyPose<TReal> pose0; pose0 = body0 ? body0->pose;
            //BodyPose<TReal> pose0; pose0 = body0 ? body0->pose;
            applyBodyPairCorrection(dx, dt);
        }
    }

private:
    /**
        * @brief Get rotated X axis.
        */
    COMM_FUNC Vector<TReal, 3> _getQuatAxis0(const Quaternion<TReal>& q) const
    {
        TReal x2 = q.x() * 2.0;
        TReal w2 = q.w() * 2.0;
        return Vector<TReal, 3>((q.w() * w2) - 1.0 + q.x() * x2, (q.z() * w2) + q.y() * x2, (-q.y() * w2) + q.z() * x2);
    }

    /**
        * @brief Get rotated Y axis.
        */
    COMM_FUNC Vector<TReal, 3> _getQuatAxis1(const Quaternion<TReal>& q) const
    {
        TReal y2 = q.y() * 2.0;
        TReal w2 = q.w() * 2.0;
        return Vector<TReal, 3>((-q.z() * w2) + q.x() * y2, (q.w() * w2) - 1.0 + q.y() * y2, (q.x() * w2) + q.z() * y2);
    }

    /**
        * @brief Get rotated Z axis.
        */
    COMM_FUNC Vector<TReal, 3> _getQuatAxis2(const Quaternion<TReal>& q) const
    {
        TReal z2 = q.z() * 2.0;
        TReal w2 = q.w() * 2.0;
        return Vector<TReal, 3>((q.y() * w2) + q.x() * z2, (-q.x() * w2) + q.y() * z2, (q.w() * w2) - 1.0 + q.z() * z2);
    }

    /**
        * @brief Apply positional or rotational correction to body0 and body1.
        * @param corr, value of correction. For rotational correction, its direction is the direction of rotation (WORLD frame), and length is rotation angle.
        * @param maxlambda, maximum lambda value, if lambda larger than this value, set lambda to 0.
        */
    COMM_FUNC TReal applyBodyPairCorrection(const Vector<TReal, 3>& corr, TReal dt, bool beposition = true, TReal maxlambda = -1.0, bool velocityLevel = false, TReal lastLambda = 0.0, bool magnitudePositive = true)
    {
        TReal C = corr.norm();
        if (C <= 0)
            return 0.0;
        Vector<TReal, 3> n(corr);
        n.normalize();
        if (!magnitudePositive)
        {
            n *= -1.0;
            C *= -1.0;
        }

        TReal w0 = body0 ? body0->getPBDInverseMass(n, this->globalPose0.position, beposition) : 0.0;
        TReal w1 = body1 ? body1->getPBDInverseMass(n, this->globalPose1.position, beposition) : 0.0;

        TReal lambda = (C * (dt * dt) + compliance * lastLambda) / ((w0 + w1) * (dt * dt) + compliance);
        //lambda = maxlambda>=0 && lambda > maxlambda ? 0 : lambda;
        lambda = maxlambda >= 0 && lambda > maxlambda ? maxlambda : lambda;

        lambda *= -1;

        n *= -lambda;

        if (body0)
            body0->applyCorrection(n, globalPose0.position, beposition, velocityLevel);
        if (body1)
        {
            n *= -1;
            body1->applyCorrection(n, globalPose1.position, beposition, velocityLevel);
        }
        return lambda;
    }

public:
    BodyInfoType* body0   = 0;
    BodyInfoType* body1   = 0;
    int           bodyId0 = 0, bodyId1 = 0;

    BodyPose<TReal> localPose0;  // Joint RELATIVE posture in body 0 local frame.
    BodyPose<TReal> localPose1;  // Joint RELATIVE posture in body 1 local frame.

    BodyPose<TReal> globalPose0;  // Joint posture (relative to body 0) in world frame.
    BodyPose<TReal> globalPose1;  // Joint postrue (reletive to body 1) in world frame.

    TReal compliance  = 0.0;
    TReal mu          = 1.0;
    TReal angDamping  = 0.0;
    TReal linDamping  = 0.0;
    TReal restitution = 1.0;

    //Vector<TReal,3>
    bool rotationXLimited;  // If X rotation is limited, this joint is a HINGE JOINT.
    bool rotationYLimited;  // Y rotation is SWING rotation.
    bool rotationZLimited;  // Z rotation is TWIST rotation.
    bool beContact = false;

    TReal minAngleX, maxAngleX;
    TReal minAngleY, maxAngleY;
    TReal minAngleZ, maxAngleZ;
    TReal maxDistance = 0.0;             // For position constraint, min distance is 0.
                                         // If max distance = 0, it's a fix position constraint;
                                         // If max distance < 0, it's a contact constraint.
    Vector<TReal, 3> normal;             // Contact joint normal, from body0 -> body1
    TReal            relVn       = 0.0;  // Normal relative velocity.
    TReal            normLambda  = 0.0;
    TReal            tangLambda0 = 0.0;
    TReal            tangLambda1 = 0.0;

    bool positionLimited = true;
    bool active          = true;
};

#ifdef PRECISION_FLOAT
//template class PBDJoint<DataType3f>;
template class PBDJoint<float>;
#else
//template class PBDJoint<DataType3d>;
template class PBDJoint<double>;

#endif

//namespace JointUtil
//{
//    COMM_FUNC void build
//}

}  // namespace PhysIKA

#endif  // PHYSIKA_PBDJOINTINFO_H