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
#include "SimpleCar.h"

bool SimpleCar::build()
{

    // Root of rigid body system node.
    m_rigidSystem = std::make_shared<RigidBodyRoot<DataType3f>>("rigid_root");
    this->addChild(m_rigidSystem);
    m_rigidSystem->setGravity(Vector3f(0, -0.9, 0));
    //m_rigidSystem->setGravity(Vector3f(0, 0, 0));

    // System state of rigid body system.
    std::shared_ptr<SystemState>       system_state = m_rigidSystem->getSystemState();
    std::shared_ptr<SystemMotionState> motion_state = system_state->m_motionState;

    // *** Rigid chassis.
    m_chassis = std::make_shared<RigidBody2<DataType3f>>("Chassis");
    m_rigidSystem->addChild(m_chassis);

    if (chassisFile != "")
    {
        m_chassis->loadShape(chassisFile);
        //Vector3f chassisMeshScale(0.3, 0.2, 0.5);
        (( std::dynamic_pointer_cast<TriangleSet<DataType3f>> )(m_chassis->getTopologyModule()))->scale(chassisMeshScale);
        (( std::dynamic_pointer_cast<TriangleSet<DataType3f>> )(m_chassis->getTopologyModule()))->translate(chassisMeshTranslate);

        //auto renderModule = std::make_shared<RigidMeshRender>(m_chassis->getTransformationFrame());
        //renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
        //m_chassis->addVisualModule(renderModule);
    }

    // Rigid inertia and position.
    m_chassis->setI(Inertia<float>(chassisMass, chassisInertia));
    m_chassis->setRelativeR(carPosition);
    m_chassis->setRelativeQ(carRotation);

    // Joint between chassis and root.
    auto rootChassisJoint = new FreeJoint("RootChassisJoint");
    m_chassis->setParentJoint(rootChassisJoint);

    //if(false)
    for (int i = 0; i < 4; ++i)
    {
        // An empty rigid for steering.
        auto steeringRigid = std::make_shared<RigidBody2<DataType3f>>("WheelSterring" + std::to_string(i));
        m_chassis->addChild(steeringRigid);
        steeringRigid->setRelativeR(wheelRelPosition[i]);
        steeringRigid->setRelativeQ(Quaternion<float>(0, 0, 0, 1));  // No relative rotation.

        if (i < 2)
            m_steeringRigid[i] = steeringRigid;

        // Joint for sterring rotation.
        auto wheelSteeringJoint = new FixedJoint("WheelSteeringJoint" + std::to_string(i));
        steeringRigid->setParentJoint(wheelSteeringJoint);
        //wheelSteeringJoint->setJointInfo(carRotation.rotate(wheelupDirection), Vector3f(0, 0, 0));
        if (i < 2)
            wheelSteeringJoint->setConstraint(0, steeringLowerBound, steeringUpperBound);
        else
            wheelSteeringJoint->setConstraint(0, 0, 0);

        // Wheel rigid.
        m_wheels[i] = std::make_shared<RigidBody2<DataType3f>>("Wheel" + std::to_string(i));
        steeringRigid->addChild(m_wheels[i]);
        if (wheelFile[i] != "")
        {
            m_wheels[i]->loadShape(wheelFile[i]);
            std::dynamic_pointer_cast<TriangleSet<DataType3f>>(m_wheels[i]->getTopologyModule())->scale(wheelMeshScale[i]);
            std::dynamic_pointer_cast<TriangleSet<DataType3f>>(m_wheels[i]->getTopologyModule())->translate(wheelMeshTranslate[i]);

            //auto renderModule = std::make_shared<RigidMeshRender>(m_wheels[i]->getTransformationFrame());
            //renderModule->setColor(Vector3f(0.8, std::rand() % 1000 / (double)1000, 0.8));
            //m_wheels[i]->addVisualModule(renderModule);
        }

        // wheel position.
        m_wheels[i]->setRelativeR(Vector3f(0, 0, 0));
        m_wheels[i]->setRelativeQ(wheelRelRotation[i].normalize());

        // Rigid mesh and inertia.
        m_wheels[i]->setI(Inertia<float>(wheelMass[i], wheelInertia[i]));

        // Joint between chassis and wheel.
        //auto chassisWheelJoint = new VehicleFrontJoint("ChassisWheelJoint" + std::to_string(i));
        //m_wheels[i]->setParentJoint(chassisWheelJoint);
        //chassisWheelJoint->setJointInfo(Vector3f(.0, 0.0, 1.0), Vector3f(0.0, 1.0, 0.0));
        //chassisWheelJoint->setConstraint(1, -1.0, 1.0);
        auto chassisWheelJoint = new FixedJoint("ChassisWheelJoint" + std::to_string(i));
        m_wheels[i]->setParentJoint(chassisWheelJoint);
        //chassisWheelJoint->setJointInfo(wheelRelRotation[i].getConjugate().rotate(wheelRightDirection), Vector3f(0.0, 0.0, 0.0));
        //chassisWheelJoint->setConstraint(0, 0, 0);
        //chassisWheelJoint->setConstraint(0, -1.0, 1.0);
    }

    forwardForcePoint = (wheelRelPosition[0] + wheelRelPosition[1]) / 2;

    /// update tree info
    m_rigidSystem->updateTree();

    return false;
}

void SimpleCar::advance(Real dt)
{
    //forward(dt);
    _updateWheelRotation(dt);
}

void SimpleCar::forward(Real dt)
{
    //Quaternion<float> chassisQuaInv = m_chassis->getGlobalQ().getConjugate();
    //Quaternion<float> qua = (chassisQuaInv * m_steeringRigid[0]->getGlobalQ() + chassisQuaInv * m_steeringRigid[1]->getGlobalQ()).normalize();

    Quaternion<float> qua = ((m_steeringRigid[0]->getGlobalQ() + m_steeringRigid[1]->getGlobalQ()) * 0.5).normalize();

    forwardDir   = qua.rotate(wheelupDirection.cross(wheelRightDirection));
    forwardForce = forwardForceAcc * (dt >= 0 ? 1 : -1);

    Vector3f force  = forwardForce * forwardDir;
    Vector3f torque = m_chassis->getGlobalQ().rotate(forwardForcePoint).cross(force);
    m_chassis->addExternalForce(force);
    m_chassis->addExternalTorque(torque);
}

void SimpleCar::backward(Real dt)
{
    forward(-dt);
}

void SimpleCar::goLeft(Real dt)
{
    currentSteering += steeringSpeed * dt;

    currentSteering = currentSteering > steeringLowerBound ? currentSteering : steeringLowerBound;
    currentSteering = currentSteering < steeringUpperBound ? currentSteering : steeringUpperBound;

    Quaternion<float> localqua(wheelupDirection, currentSteering);

    Quaternion<float> frontwheelqua = /*m_chassis->getGlobalQ() * */ localqua;

    m_steeringRigid[0]->setRelativeQ(frontwheelqua);
    m_steeringRigid[1]->setRelativeQ(frontwheelqua);
}

void SimpleCar::goRight(Real dt)
{
    goLeft(-dt);
}

class SetTimeAction : public Action
{
public:
    void process(Node* node)
    {
        node->setDt(dt);
    }

    float dt = 0.016;
};
void SimpleCar::setDt(Real dt)
{
    Node::setDt(dt);
    SetTimeAction dtime;
    dtime.dt = dt;
    m_rigidSystem->traverseTopDown(&dtime);
}

void SimpleCar::_updateWheelRotation(Real dt)
{
    return;
    Vector3f forwarDir = wheelupDirection.cross(wheelRightDirection);
    forwarDir          = m_chassis->getGlobalQ().rotate(forwarDir);

    for (int i = 0; i < 4; ++i)
    {
        Vector3f wheelV     = m_wheels[i]->getLinearVelocity();
        float    wheelVnorm = wheelV.norm();
        float    angv       = wheelVnorm / wheelRadius[i];
        float    flag       = wheelV.dot(forwarDir) > 0 ? -1.0 : 1.0;

        Quaternion<float> qua(angv * dt * flag, wheelRightDirection);
        qua = m_wheels[i]->getRelativeQ() * qua;
        m_wheels[i]->setRelativeQ(qua);
    }
}
