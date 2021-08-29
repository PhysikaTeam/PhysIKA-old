#pragma once

#include "GUI/GlutGUI/GLApp.h"
#include "Framework/Framework/Node.h"

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"

//using namespace PhysIKA;

namespace PhysIKA {
template <int N = 4>
class MultiWheelCar : public Node
{
public:
    MultiWheelCar() {}

    bool build();

    //virtual bool initialize() override;

    virtual void advance(Real dt);

    void forward(Real dt);

    void backward(Real dt);

    void goLeft(Real dt);

    void goRight(Real dt);

    //void setDt(Real dt);

    std::shared_ptr<RigidBody2<DataType3f>> getChassis()
    {
        return m_chassis;
    }
    std::shared_ptr<RigidBody2<DataType3f>> getWheels(int i)
    {
        return m_wheels[i];
    }

    void updateForce(Real dt);

private:
    void _updateWheelRotation(Real dt);

    Vector3f _getForwardDir();

    Vector3f _getRightDir();

    Vector3f _getUpDir();

    void _setRigidForceAsGravity();

    void _doVelConstraint(Real dt);

    Quaternionf _rotationToStandardLocal();

public:
    // Position and rotation of car.
    Vector3f          carPosition;
    Quaternion<float> carRotation;

    Vector3f wheelRelPos[2][N];  // Relative position of left wheels.

    Quaternion<float> wheelRelRot[2][N];  // Relative rotation of left wheels.

    Vector3f upDirection;     // Up direction in car local frame.
    Vector3f rightDirection;  // Right direction in car local frame.

    // Visualization information.
    bool        needVisualization = true;
    std::string chassisFile       = "";

    std::string wheelFile[2][N];

    Vector3f chassisMeshScale;
    Vector3f wheelMeshScale[2][N];

    Vector3f chassisMeshTranslate;
    Vector3f wheelMeshTranslate[2][N];

    float    chassisMass = 1.0;
    Vector3f chassisInertia;
    float    wheelMass[2][N];
    Vector3f wheelInertia[2][N];

    Vector3f m_gravity = { 0, -9.8, 0 };

    std::shared_ptr<PBDSolver>              m_rigidSolver;
    std::shared_ptr<RigidBody2<DataType3f>> m_chassis;

    //std::shared_ptr<RigidBody2<DataType3f>>m_steeringRigid[N];
    std::shared_ptr<RigidBody2<DataType3f>> m_wheels[2][N];
    std::shared_ptr<RigidBody2<DataType3f>> m_steeringRigid[2][N];

    int chassisCollisionGroup = 1;
    int chassisCollisionMask  = 1;
    int wheelCollisionGroup   = 1;
    int wheelCollisionMask    = 1;

    float forwardForceAcc;

    float maxVel = 2.5;

    float linearDamping  = 0;
    float angularDamping = 0;

    float suspensionLength   = 0.05;
    float suspensionStrength = 1000000;

private:
    Vector3f lforwardForcePoint;
    Vector3f rforwardForcePoint;

    Vector3f forwardDir;
    float    lforwardForce = 0;
    float    rforwardForce = 0;

    float breakForce = 0;

    bool m_accPressed = false;

    //float m_curSuspensionExt[0]
};

template <int N>
bool MultiWheelCar<N>::build()
{
    // *** Rigid chassis.
    m_chassis = std::make_shared<RigidBody2<DataType3f>>("Chassis");
    this->addChild(m_chassis);

    if (chassisFile != "")
    {
        m_chassis->loadShape(chassisFile);
        //Vector3f chassisMeshScale(0.3, 0.2, 0.5);
        (( std::dynamic_pointer_cast<TriangleSet<DataType3f>> )(m_chassis->getTopologyModule()))->scale(chassisMeshScale);
        (( std::dynamic_pointer_cast<TriangleSet<DataType3f>> )(m_chassis->getTopologyModule()))->translate(chassisMeshTranslate);
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

    for (int lr = 0; lr < 2; ++lr)
    {
        int idwheels[N];
        for (int i = 0; i < N; ++i)
        {
            // wheel position.
            Vector3f    wheelPos = carPosition + carRotation.rotate(wheelRelPos[lr][i]);
            Quaternionf wheelRot = carRotation * wheelRelRot[lr][i];

            // steering rigid.
            auto steeringRigid = std::make_shared<RigidBody2<DataType3f>>("WheelSterring_" + std::to_string(lr) + std::string("_") + std::to_string(i));
            m_chassis->addChild(steeringRigid);
            int idSteering = m_rigidSolver->addRigid(steeringRigid);

            steeringRigid->setGlobalR(wheelPos);
            steeringRigid->setGlobalQ(wheelRot);
            steeringRigid->setI(Inertia<float>(wheelMass[lr][i], wheelInertia[lr][i]));
            steeringRigid->setExternalForce(Vector3f(0.0, -9.8 * wheelMass[lr][i], 0.0));
            steeringRigid->setLinearDamping(linearDamping);
            steeringRigid->setAngularDamping(angularDamping);

            // Collision filter.
            steeringRigid->setCollisionFilterGroup(0);
            steeringRigid->setCollisionFilterMask(0);

            m_steeringRigid[lr][i] = steeringRigid;

            // Steering joint.
            PBDJoint<double> steeringJoint;
            steeringJoint.bodyId0 = idchassis;
            steeringJoint.bodyId1 = idSteering;

            Vector3d sjointOffset(upDirection[0], upDirection[1], upDirection[2]);
            sjointOffset *= suspensionLength * 0.5;

            Vector3f wheelRelPosition         = wheelRelPos[lr][i];
            steeringJoint.localPose0.position = sjointOffset + Vector3d(wheelRelPosition[0], wheelRelPosition[1], wheelRelPosition[2]);
            steeringJoint.localPose0.rotation = localToStd.getConjugate() /* * Quaterniond(0, 0, -1, 1).normalize()*/;
            steeringJoint.localPose1.position = Vector3d(0, 0, 0);
            Quaternionf tmpq                  = wheelRot.getConjugate() * carRotation;
            steeringJoint.localPose1.rotation = (Quaterniond(tmpq[0], tmpq[1], tmpq[2], tmpq[3]) * steeringJoint.localPose0.rotation).normalize();
            steeringJoint.compliance          = 0.000000;

            steeringJoint.rotationYLimited = true;
            steeringJoint.minAngleY        = 0;
            steeringJoint.maxAngleY        = 0;
            steeringJoint.rotationXLimited = true;
            steeringJoint.minAngleX        = 0;
            steeringJoint.maxAngleX        = 0;
            steeringJoint.rotationZLimited = false;
            steeringJoint.beContact        = false;
            steeringJoint.positionLimited  = true;
            steeringJoint.maxDistance      = suspensionLength;

            m_rigidSolver->addPBDJoint(steeringJoint, idchassis, idSteering);

            // Wheel rigid.
            m_wheels[lr][i] = std::make_shared<RigidBody2<DataType3f>>("Wheel_" + std::to_string(lr) + std::string("_") + std::to_string(i));
            m_wheels[lr][i]->setMu(5.0);
            steeringRigid->addChild(m_wheels[lr][i]);
            int idwheel = m_rigidSolver->addRigid(m_wheels[lr][i]);
            idwheels[i] = idwheel;
            if (wheelFile[lr][i] != "")
            {
                m_wheels[lr][i]->loadShape(wheelFile[lr][i]);
                auto pwheelTopology = std::dynamic_pointer_cast<TriangleSet<DataType3f>>(m_wheels[lr][i]->getTopologyModule());
                pwheelTopology->scale(wheelMeshScale[lr][i]);
                pwheelTopology->translate(wheelMeshTranslate[lr][i]);
            }

            m_wheels[lr][i]->setGlobalR(wheelPos);
            m_wheels[lr][i]->setGlobalQ(wheelRot);
            //if (i < 2)
            //{
            //    wheelLocalRight[i] = wheelRot.getConjugate().rotate(
            //        carRotation.rotate(wheelRightDirection)
            //    );
            //}

            // Rigid mesh and inertia.
            m_wheels[lr][i]->setI(Inertia<float>(wheelMass[lr][i], wheelInertia[lr][i]));
            m_wheels[lr][i]->setExternalForce(Vector3f(0, -9.8 * wheelMass[lr][i], 0));

            m_wheels[lr][i]->setLinearDamping(linearDamping);
            m_wheels[lr][i]->setAngularDamping(angularDamping);

            // Collision filter.
            m_wheels[lr][i]->setCollisionFilterGroup(wheelCollisionGroup);
            m_wheels[lr][i]->setCollisionFilterMask(wheelCollisionMask);

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
    }

    //forwardForcePoint = (wheelRelPosition[0] + wheelRelPosition[1]) / 2;
    lforwardForcePoint = wheelRelPos[0][0];
    rforwardForcePoint = wheelRelPos[1][0];

    m_chassis->setLinearVelocity(m_chassis->getLinearVelocity() * 0.99);
    m_rigidSolver->addCustomUpdateFunciton(std::bind(&MultiWheelCar<N>::updateForce, this, std::placeholders::_1));

    return true;
}

template <int N>
void MultiWheelCar<N>::advance(Real dt)
{

    //this->updateForce(dt);
    if (!m_accPressed)
    {
        lforwardForce = 0;
        rforwardForce = 0;
    }
    m_accPressed = false;

    //this->_updateWheelRotation(dt);
    this->_doVelConstraint(dt);
    this->m_rigidSolver->setBodyDirty();

    //Vector3f carVel = m_chassis->getLinearVelocity();
    //printf("Car Vel:  %lf %lf %lf \n", carVel[0], carVel[1], carVel[2]);

    return;
}

template <int N>
void MultiWheelCar<N>::forward(Real dt)
{
    forwardDir = this->_getForwardDir();

    float dforce = forwardForceAcc * (dt >= 0 ? 1 : -1);

    lforwardForce = (lforwardForce + rforwardForce) / 2.0;
    rforwardForce = lforwardForce;

    lforwardForce += dforce;
    rforwardForce += dforce;

    //Vector3f torque = m_chassis->getGlobalQ().rotate(forwardForcePoint).cross(force);
    //m_chassis->addExternalForce(force);
    //m_chassis->addExternalTorque(torque);

    m_accPressed = true;
}

template <int N>
void MultiWheelCar<N>::backward(Real dt)
{
    forward(-dt);
}

template <int N>
void MultiWheelCar<N>::goLeft(Real dt)
{
    float dforce = forwardForceAcc * (dt >= 0 ? 1 : -1);
    rforwardForce += dforce;
    lforwardForce -= dforce;

    //lforwardForce = 0;

    m_accPressed = true;
}

template <int N>
void MultiWheelCar<N>::goRight(Real dt)
{
    float dforce = forwardForceAcc * (dt >= 0 ? 1 : -1);
    lforwardForce += dforce;
    rforwardForce -= dforce;

    //rforwardForce = 0;
    m_accPressed = true;
}

//template<int N>
//void MultiWheelCar<N>::_updateWheelRotation(Real dt)
//{
//    Quaternionf relq0 = m_steeringRigid[0]->getGlobalQ().getConjugate() * m_wheels[0]->getGlobalQ();
//    Quaternionf relq1 = m_steeringRigid[1]->getGlobalQ().getConjugate() * m_wheels[1]->getGlobalQ();

//    currentSteering = currentSteering > steeringLowerBound ? currentSteering : steeringLowerBound;
//    currentSteering = currentSteering < steeringUpperBound ? currentSteering : steeringUpperBound;

//    //Vector3f gloUp = m_chassis->getGlobalQ().rotate(wheelupDirection);
//    Quaternion<float> localqua(wheelupDirection, currentSteering);

//    m_steeringRigid[0]->setGlobalQ(m_chassis->getGlobalQ() *localqua* wheelRelRotation[0]);
//    m_steeringRigid[1]->setGlobalQ(m_chassis->getGlobalQ() *localqua* wheelRelRotation[1]);
//
//    m_wheels[0]->setGlobalQ(m_steeringRigid[0]->getGlobalQ() * relq0);
//    m_wheels[1]->setGlobalQ(m_steeringRigid[1]->getGlobalQ() * relq1);
//
//}

template <int N>
Vector3f MultiWheelCar<N>::_getForwardDir()
{
    Vector3f curRight = _getRightDir();
    Vector3f curUp    = _getUpDir();
    forwardDir        = curUp.cross(curRight).normalize();

    return forwardDir;
}

template <int N>
Vector3f MultiWheelCar<N>::_getRightDir()
{
    return m_chassis->getGlobalQ().rotate(rightDirection);
}

template <int N>
Vector3f MultiWheelCar<N>::_getUpDir()
{
    return m_chassis->getGlobalQ().rotate(upDirection);
}

//template<int N>
//void MultiWheelCar<N>::_setRigidForceAsGravity()
//{
//    m_chassis->setExternalForce(m_gravity * chassisMass);
//    m_chassis->setExternalTorque(Vector3f());

//    for (int i = 0; i < N; ++i)
//    {
//        m_wheels[i]->setExternalForce(m_gravity * wheelMass[i]);
//        m_wheels[i]->setExternalTorque(Vector3f());

//        m_steeringRigid[i]->setExternalForce(m_gravity * wheelMass[i]);
//        m_steeringRigid[i]->setExternalTorque(Vector3f());
//    }
//}

template <int N>
void MultiWheelCar<N>::_doVelConstraint(Real dt)
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

template <int N>
Quaternionf MultiWheelCar<N>::_rotationToStandardLocal()
{
    // updir <==> (0, 1, 0)
    // rightdir <==> (1, 0, 0)

    Vector3f stdup(0, 1, 0);
    Vector3f stdright(1, 0, 0);

    Quaternionf rot;

    Vector3f axisUp = upDirection.cross(stdup);
    if (axisUp.norm() > 1e-3)
    {
        axisUp.normalize();

        float rad = upDirection.dot(stdup);
        rad       = acos(rad);
        rot       = Quaternionf(axisUp, rad);
    }

    Vector3f axisRight = rightDirection.cross(stdright);
    if (axisRight.norm() > 1e-3)
    {
        axisRight.normalize();
        float rad = rightDirection.dot(stdright);
        rad       = acos(rad);
        rot       = Quaternionf(axisRight, rad) * rot;
    }

    return rot;
}

template <int N>
void MultiWheelCar<N>::updateForce(Real dt)
{
    Vector3f rightdir = _getRightDir();

    Vector3f chaForce;
    Vector3f chaTorque;
    for (int lr = 0; lr < 2; ++lr)
    {
        for (int i = 0; i < N; ++i)
        {
            Vector3f relpglo = m_steeringRigid[lr][i]->getGlobalR() - m_chassis->getGlobalR();
            Vector3f relp    = m_chassis->getGlobalQ().getConjugate().rotate(relpglo);

            relp         = relp - wheelRelPos[lr][i];
            float curlen = relp.norm();

            Vector3f force = -suspensionStrength * relp;
            m_chassis->getGlobalQ().rotate(force);

            m_steeringRigid[lr][i]->setExternalForce(force + m_gravity * m_steeringRigid[lr][i]->getI().getMass());

            ////if (abs(lforwardForce) < abs(rforwardForce) - 1 && lr == 0)
            ////{
            ////    m_wheels[lr][i]->setAngularVelocity(Vector3f());
            ////}
            ////else if (abs(rforwardForce) < abs(lforwardForce) - 1 && lr == 1)
            ////{
            ////    m_wheels[lr][i]->setAngularVelocity(Vector3f());
            ////}

            float    wheelT = lr == 0 ? -lforwardForce : -rforwardForce;
            Vector3f vT     = rightdir * wheelT;
            m_wheels[lr][i]->setExternalTorque(vT);

            Vector3f chaF = -force;
            Vector3f chaT = relpglo.cross(chaF);
            chaForce += chaF;
            chaTorque += chaT;
        }
    }

    //forwardDir = this->_getForwardDir();
    //Vector3f lforwardF = forwardDir * lforwardForce;
    //Vector3f lforwardT = m_chassis->getGlobalQ().rotate(lforwardForcePoint).cross(lforwardF);
    //chaForce += lforwardF;
    //chaTorque += lforwardT;

    //Vector3f rforwardF = forwardDir * rforwardForce;
    //Vector3f rforwardT = m_chassis->getGlobalQ().rotate(rforwardForcePoint).cross(rforwardF);
    //chaForce += rforwardF;
    //chaTorque += rforwardT;

    chaForce += m_gravity * m_chassis->getI().getMass();
    m_chassis->setExternalForce(chaForce);
    m_chassis->setExternalTorque(chaTorque);
}

}  // namespace PhysIKA