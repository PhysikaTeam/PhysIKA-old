#include "Joint.h"
#include "Dynamics/RigidBody/RigidUtil.h"
namespace PhysIKA {
//IMPLEMENT_CLASS_1(Joint, TDataType)

//PhysIKA::Joint ::Joint(std::string name):
//    m_XT(6,6), m_XJ(6,6),m_fa(6), m_r(3)
//{
//    //attachField(&m_predecessor, MechanicalState::mass(), "Total mass of the rigid body!", false);
//    m_XT.identity();
//    m_XJ.identity();
//}

//Joint ::Joint(Node* predecessor, Node* successor)
//{
//    m_predecessor = predecessor;
//    m_successor = successor;
//}

//Joint ::~Joint()
//{

//}

//void Joint ::setRigidBody(Node * predecessor, Node * successor)
//{
//    m_predecessor = predecessor;
//    m_successor = successor;
//}

//template<unsigned int Dof>
//void Joint ::setConstrainedSpaceMatrix(const JointSpace<float, Dof>& s)
//{
//    m_S = s;
//    m_constrain_dim = s.cols();
//    _updateT();

//    this->m_dq.resize(s.cols());
//    this->m_ddq.resize(s.cols());
//}

//
//void Joint::update(double dt)
//{

//}

//void Joint ::_updateT()
//{
//    RigidUtil::calculateOrthogonalSpace(this->m_S, this->m_T);
//}

//
//bool Joint ::initialize()
//{
//    return true;
//}

}  // namespace PhysIKA