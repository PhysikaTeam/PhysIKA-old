#include "Dynamics/RigidBody/PBDRigid/PBDSolverNode.h"
#include "Core/Utility/Function1Pt.h"

#include <iostream>
#include <Windows.h>

//namespace PhysIKA
//{
//	//IMPLEMENT_CLASS_1(PBDSolverNode, TReal)
//}

namespace PhysIKA {
//template<typename TReal>
PBDSolverNode::PBDSolverNode()
{
    m_solver = std::make_shared<PBDSolver>();
}

//template<typename TReal>
bool PBDSolverNode::initialize()
{
    if (m_solver)
        return m_solver->initialize();
    return true;
}

//template<typename TReal>
void PBDSolverNode::advance(Real dt)
{

    if (m_needForward)
        m_solver->advance(dt);
}

//template<typename TReal>
int PBDSolverNode::addRigid(RigidBody2_ptr prigid)
{
    if (prigid)
    {
        this->addChild(prigid);
        return m_solver->addRigid(prigid);
    }
    return -1;
}

//template<typename TReal>
int PBDSolverNode::addPBDJoint(const PBDJoint<double>& pbdjoint, int bodyid0, int bodyid1)
{
    return m_solver->addPBDJoint(pbdjoint, bodyid0, bodyid1);
}

}  // namespace PhysIKA