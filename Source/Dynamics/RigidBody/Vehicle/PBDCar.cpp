//#include "demoCar.h"

#include "Dynamics/ParticleSystem/StaticBoundary.h"
#include "Dynamics/RigidBody/FreeJoint.h"
//#include "Rendering/RigidMeshRender.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "VehicleFrontJoint.h"
#include "VehicleRearJoint.h"
#include "Dynamics/RigidBody/FixedJoint.h"
#include "Framework/Action/Action.h"
#include <string>
#include "PBDCar.h"
#include <functional>
#include <cmath>
namespace PhysIKA {
bool PBDCar::build()
{
    // *** Rigid chassis.
    m_chassis = std::make_shared<RigidBody2<DataType3f>>("Chassis");
    this->addChild(m_chassis);

    if (chassisFile != "")
    {
        m_chassis->loadShape(chassisFile);
        //Vector3f chassisMeshScale(0.3, 0.2, 0.5);
        ((std::dynamic_pointer_cast<TriangleSet<DataType3f>>) (m_chassis->getTopologyModule()))->scale(chassisMeshScale);
        ((std::dynamic_pointer_cast<TriangleSet<DataType3f>>) (m_chassis->getTopologyModule()))->translate(chassisMeshTranslate);
    }

    // Rigid inertia and position.
    m_chassis->setI(Inertia<float>(chassisMass, chassisInertia));
    m_chassis->setGlobalR(carPosition);
    m_chassis->setGlobalQ(carRotation);
    m_chassis->setExternalForce(Vector3f(0.0 * chassisMass, -9.8 * chassisMass, 0));
    m_chassis->setLinearDamping(linearDamping);
    m_chassis->setAngularDamping(angularDamping);

    // Collision filter.
    m_chassis->setCollisionFilterGroup(chassisCollisionGroup);
    m_chassis->setCollisionFilterMask(chassisCollisionMask);

    int idchassis = m_rigidSolver->addRigid(m_chassis);

    //return true;
    Quaternionf localToStd_ = _rotationToStandardLocal();
    Quaterniond localToStd(localToStd_.x(), localToStd_.y(), localToStd_.z(), localToStd_.w());

    int idwheels[4];
    for (int i = 0; i < 4; ++i)
    {
        // wheel position.
        Vector3f    wheelPos = carPosition + carRotation.rotate(wheelRelPosition[i]);
        Quaternionf wheelRot = carRotation * wheelRelRotation[i];

        // steering rigid.
        auto steeringRigid = std::make_shared<RigidBody2<DataType3f>>("WheelSterring" + std::to_string(i));
        m_chassis->addChild(steeringRigid);
        int idSteering = m_rigidSolver->addRigid(steeringRigid);

        steeringRigid->setGlobalR(wheelPos);
        steeringRigid->setGlobalQ(wheelRot);
        steeringRigid->setI(Inertia<float>(wheelMass[i], wheelInertia[i]));
        steeringRigid->setExternalForce(Vector3f(0.0, -9.8 * wheelMass[i], 0.0));
        steeringRigid->setLinearDamping(linearDamping);
        steeringRigid->setAngularDamping(angularDamping);

        // Collision filter.
        steeringRigid->setCollisionFilterGroup(0);
        steeringRigid->setCollisionFilterMask(0);

        m_steeringRigid[i] = steeringRigid;

        // Steering joint.
        PBDJoint<double> steeringJoint;
        steeringJoint.bodyId0 = idchassis;
        steeringJoint.bodyId1 = idSteering;

        Vector3d sjointOffset(wheelupDirection[0], wheelupDirection[1], wheelupDirection[2]);
        sjointOffset *= suspensionLength * 0.5;

        steeringJoint.localPose0.position = sjointOffset + Vector3d(wheelRelPosition[i][0], wheelRelPosition[i][1], wheelRelPosition[i][2]);
        steeringJoint.localPose0.rotation = localToStd.getConjugate() /* * Quaterniond(0, 0, -1, 1).normalize()*/;
        steeringJoint.localPose1.position = Vector3d(0, 0, 0);
        Quaternionf tmpq                  = wheelRot.getConjugate() * carRotation;
        steeringJoint.localPose1.rotation = (Quaterniond(tmpq[0], tmpq[1], tmpq[2], tmpq[3]) * steeringJoint.localPose0.rotation).normalize();
        steeringJoint.compliance          = 0.000000;

        steeringJoint.rotationYLimited = true;
        steeringJoint.minAngleY        = 0;
        steeringJoint.maxAngleY        = 0;
        steeringJoint.rotationXLimited = true;
        if (i < 2)
        {
            steeringJoint.minAngleX = steeringLowerBound;
            steeringJoint.maxAngleX = steeringUpperBound;
        }
        else
        {
            steeringJoint.minAngleX = 0;
            steeringJoint.maxAngleX = 0;
        }

        steeringJoint.rotationZLimited = false;
        steeringJoint.beContact        = false;
        steeringJoint.positionLimited  = true;

        steeringJoint.maxDistance = suspensionLength;

        m_rigidSolver->addPBDJoint(steeringJoint, idchassis, idSteering);

        // Wheel rigid.
        m_wheels[i] = std::make_shared<RigidBody2<DataType3f>>("Wheel" + std::to_string(i));
        m_wheels[i]->setMu(5.0);
        steeringRigid->addChild(m_wheels[i]);
        int idwheel = m_rigidSolver->addRigid(m_wheels[i]);
        idwheels[i] = idwheel;
        if (wheelFile[i] != "")
        {
            m_wheels[i]->loadShape(wheelFile[i]);
            std::dynamic_pointer_cast<TriangleSet<DataType3f>>(m_wheels[i]->getTopologyModule())->scale(wheelMeshScale[i]);
            std::dynamic_pointer_cast<TriangleSet<DataType3f>>(m_wheels[i]->getTopologyModule())->translate(wheelMeshTranslate[i]);
        }

        m_wheels[i]->setGlobalR(wheelPos);
        m_wheels[i]->setGlobalQ(wheelRot);
        if (i < 2)
        {
            wheelLocalRight[i] = wheelRot.getConjugate().rotate(
                carRotation.rotate(wheelRightDirection));
        }

        // Rigid mesh and inertia.
        m_wheels[i]->setI(Inertia<float>(wheelMass[i], wheelInertia[i]));
        m_wheels[i]->setExternalForce(Vector3f(0, -9.8 * wheelMass[i], 0));

        m_wheels[i]->setLinearDamping(linearDamping);
        m_wheels[i]->setAngularDamping(angularDamping);

        // Collision filter.
        m_wheels[i]->setCollisionFilterGroup(wheelCollisionGroup);
        m_wheels[i]->setCollisionFilterMask(wheelCollisionMask);

        // Wheel joint definition.
        PBDJoint<double> wheelJoint;
        wheelJoint.bodyId0 = idSteering;
        wheelJoint.bodyId1 = idwheel;

        wheelJoint.localPose0.position = steeringJoint.localPose1.position;
        wheelJoint.localPose0.rotation = steeringJoint.localPose1.rotation * Quaterniond(0, 0, -1, 1).normalize();
        wheelJoint.localPose1.position = steeringJoint.localPose1.position;
        wheelJoint.localPose1.rotation = steeringJoint.localPose1.rotation * Quaterniond(0, 0, -1, 1).normalize();
        wheelJoint.compliance          = 0.0000001;

        wheelJoint.rotationXLimited = false;
        wheelJoint.rotationYLimited = true;
        wheelJoint.minAngleY        = 0;
        wheelJoint.maxAngleY        = 0;
        wheelJoint.rotationZLimited = false;
        wheelJoint.beContact        = false;
        wheelJoint.positionLimited  = true;

        m_rigidSolver->addPBDJoint(wheelJoint, idSteering, idwheel);
    }

    forwardForcePoint = (wheelRelPosition[0] + wheelRelPosition[1]) / 2;

    m_chassis->setLinearVelocity(m_chassis->getLinearVelocity() * 0.99);
    m_rigidSolver->addCustomUpdateFunciton(std::bind(&PBDCar::updateForce, this, std::placeholders::_1));

    return true;
}

void PBDCar::advance(Real dt)
{
    //return;
    //forward(dt);
    //_updateWheelRotation(dt);

    //this->updateForce(dt);
    if (!m_accPressed)
    {
        forwardForce = 0;
    }
    m_accPressed = false;

    this->_updateWheelRotation(dt);
    this->_doVelConstraint(dt);
    this->m_rigidSolver->setBodyDirty();

    //Vector3f carVel = m_chassis->getLinearVelocity();
    //printf("Car Vel:  %lf %lf %lf \n", carVel[0], carVel[1], carVel[2]);

    return;
}

void PBDCar::forward(Real dt)
{
    forwardDir = this->_getForwardDir();
    forwardForce += forwardForceAcc * (dt >= 0 ? 1 : -1);

    //Vector3f force = forwardForce * forwardDir;
    //Vector3f torque = m_chassis->getGlobalQ().rotate(forwardForcePoint).cross(force);
    //m_chassis->addExternalForce(force);
    //m_chassis->addExternalTorque(torque);

    m_accPressed = true;
}

void PBDCar::backward(Real dt)
{
    forward(-dt);
}

void PBDCar::goLeft(Real dt)
{
    currentSteering += steeringSpeed * dt;

    currentSteering = currentSteering > steeringLowerBound ? currentSteering : steeringLowerBound;
    currentSteering = currentSteering < steeringUpperBound ? currentSteering : steeringUpperBound;

    m_accPressed = true;
}

void PBDCar::goRight(Real dt)
{
    goLeft(-dt);
}

void PBDCar::_updateWheelRotation(Real dt)
{
    Quaternionf relq0 = m_steeringRigid[0]->getGlobalQ().getConjugate() * m_wheels[0]->getGlobalQ();
    Quaternionf relq1 = m_steeringRigid[1]->getGlobalQ().getConjugate() * m_wheels[1]->getGlobalQ();

    currentSteering = currentSteering > steeringLowerBound ? currentSteering : steeringLowerBound;
    currentSteering = currentSteering < steeringUpperBound ? currentSteering : steeringUpperBound;

    //Vector3f gloUp = m_chassis->getGlobalQ().rotate(wheelupDirection);
    Quaternion<float> localqua(wheelupDirection, currentSteering);

    m_steeringRigid[0]->setGlobalQ(m_chassis->getGlobalQ() * localqua * wheelRelRotation[0]);
    m_steeringRigid[1]->setGlobalQ(m_chassis->getGlobalQ() * localqua * wheelRelRotation[1]);

    m_wheels[0]->setGlobalQ(m_steeringRigid[0]->getGlobalQ() * relq0);
    m_wheels[1]->setGlobalQ(m_steeringRigid[1]->getGlobalQ() * relq1);
}

Vector3f PBDCar::_getForwardDir()
{
    Vector3f curRight = m_wheels[0]->getGlobalQ().rotate(wheelLocalRight[0]);
    Vector3f curUp    = m_chassis->getGlobalQ().rotate(wheelupDirection);
    forwardDir        = curUp.cross(curRight).normalize();

    return forwardDir;
}

Vector3f PBDCar::_getRightDir()
{
    return m_wheels[0]->getGlobalQ().rotate(wheelLocalRight[0]);
}

Vector3f PBDCar::_getUpDir()
{
    return m_chassis->getGlobalQ().rotate(wheelupDirection);
}

void PBDCar::_setRigidForceAsGravity()
{
    m_chassis->setExternalForce(m_gravity * chassisMass);
    m_chassis->setExternalTorque(Vector3f());

    for (int i = 0; i < 4; ++i)
    {
        m_wheels[i]->setExternalForce(m_gravity * wheelMass[i]);
        m_wheels[i]->setExternalTorque(Vector3f());

        m_steeringRigid[i]->setExternalForce(m_gravity * wheelMass[i]);
        m_steeringRigid[i]->setExternalTorque(Vector3f());
    }
}

void PBDCar::_doVelConstraint(Real dt)
{
    double linDamp = pow(1.0 - linearDamping, dt);
    double angDamp = pow(1.0 - angularDamping, dt);

    m_chassis->setLinearVelocity(m_chassis->getLinearVelocity() * linDamp);
    m_chassis->setAngularVelocity(m_chassis->getAngularVelocity() * angDamp);

    if (m_chassis->getLinearVelocity().norm() > maxVel)
    {
        double fac = maxVel / m_chassis->getLinearVelocity().norm();
        m_chassis->setLinearVelocity(m_chassis->getLinearVelocity() * fac);
    }
}

Quaternionf PBDCar::_rotationToStandardLocal()
{
    // updir <==> (0, 1, 0)
    // rightdir <==> (1, 0, 0)

    Vector3f stdup(0, 1, 0);
    Vector3f stdright(1, 0, 0);

    Quaternionf rot;

    Vector3f axisUp = wheelupDirection.cross(stdup);
    if (axisUp.norm() > 1e-3)
    {
        axisUp.normalize();

        float rad = wheelupDirection.dot(stdup);
        rad       = acos(rad);
        rot       = Quaternionf(axisUp, rad);
    }

    Vector3f axisRight = wheelRightDirection.cross(stdright);
    if (axisRight.norm() > 1e-3)
    {
        axisRight.normalize();
        float rad = wheelRightDirection.dot(stdright);
        rad       = acos(rad);
        rot       = Quaternionf(axisRight, rad) * rot;
    }

    return rot;
}

void PBDCar::updateForce(Real dt)
{
    Vector3f chaForce;
    Vector3f chaTorque;
    for (int i = 0; i < 4; ++i)
    {
        Vector3f relpglo = m_steeringRigid[i]->getGlobalR() - m_chassis->getGlobalR();
        Vector3f relp    = m_chassis->getGlobalQ().getConjugate().rotate(relpglo);

        relp         = relp - wheelRelPosition[i];
        float curlen = relp.norm();

        Vector3f force = -suspensionStrength * relp;

        m_steeringRigid[i]->setExternalForce(force + m_gravity * m_steeringRigid[i]->getI().getMass());

        Vector3f chaF = -force;
        Vector3f chaT = relpglo.cross(chaF);
        chaForce += chaF;
        chaTorque += chaT;
    }

    forwardDir        = this->_getForwardDir();
    Vector3f forwardF = forwardDir * forwardForce;
    Vector3f forwardT = m_chassis->getGlobalQ().rotate(forwardForcePoint).cross(forwardF);

    chaForce += forwardF;
    chaTorque += forwardT;

    chaForce += m_gravity * m_chassis->getI().getMass();
    m_chassis->setExternalForce(chaForce);
    m_chassis->setExternalTorque(chaTorque);
}

}  // namespace PhysIKA