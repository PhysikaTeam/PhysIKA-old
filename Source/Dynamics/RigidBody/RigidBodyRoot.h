#pragma once

#include "Dynamics/RigidBody/RigidBody2.h"
#include "Dynamics/RigidBody/Joint.h"
#include "Dynamics/RigidBody/SystemState.h"

#include <list>

namespace PhysIKA {
//template<typename TDataType> class Frame;
/*!
    *    \class    RigidBody
    *    \brief    Rigid body dynamics.
    *
    *    This class implements a simple rigid body.
    *
    */
template <typename TDataType>
class RigidBodyRoot : public RigidBody2<TDataType>
{
    DECLARE_CLASS_1(RigidBodyRoot, TDataType)
public:
    //typedef typename TDataType::Real Real;
    //typedef typename TDataType::Coord Coord;
    //typedef typename TDataType::Matrix Matrix;

    RigidBodyRoot(std::string name = "default");
    virtual ~RigidBodyRoot();

    void advance(Real dt) override;
    void updateTopology() override;

    void addLoopJoint(Joint* joint);
    void addLoopJoint(std::shared_ptr<Joint> joint);

    //MatrixMN<float> accumulateMatrices(MatrixMN<float>& P, MatrixMN<float>& I, MatrixMN<float>& T);

    bool initialize() override;

    // get all descendant nodes
    virtual const std::vector<std::shared_ptr<RigidBody2<TDataType>>>& getAllNode() const
    {
        return m_all_childs_nodes;
    }

    // get all descendant nodes in pair
    // pair: first: parent id in the vector; second: shared_ptr of node
    virtual const std::vector<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>>& getAllParentidNodePair() const
    {
        return m_all_childs_node_pairs;
    }

    //virtual SpatialVector<float> calMomentum6();
    //virtual Vector3f calMomentum();
    //virtual Vector3f calAngularMomentum();
    //virtual float calKineticEnergy();
    //virtual void calEnergyAndMomentum(Vector3f& m, float& e);

    // update tree topology info
    void updateTree();

    void initSystemState();

    const Vector3f& getGravity() const
    {
        return m_state->m_gravity;
    }
    void setGravity(const Vector3f& g)
    {
        m_state->m_gravity = g;
    }

    int getJointDof() const
    {
        return m_joint_dof;
    }
    const std::vector<int>& getJointIdxMap() const
    {
        return m_joint_idx_map;
    }

    std::shared_ptr<SystemState> getSystemState()
    {
        return m_state;
    }
    void setSystemState(std::shared_ptr<SystemState> state)
    {
        m_state = state;
    }
    //cosnt SystemState& getSystemState() const { return m_state; }

    //void applyExternalForce()
    void collectForceState();

private:
    void _updateTreeGlobalInfo();

    void _collectMotionState();

    void _collectRelativeMotionState();

    void _clearRIgidForce();

    int _calculateDof();

    //void _getAccelerationQ(std::vector<Vectornd<float>>& ddq);

private:
    // list of loop joint
    // At present, loop joints are not allowed
    ListPtr<Joint> m_loopJoints;

    // total degrees of freedom of all joints.
    int m_joint_dof = 0;

    // idx_map
    // as the freedom of degree of every joint is different, we need to find which elements in C & H is associate with current joint
    // idx_map maps index i to the start index of elements of i-th node.
    std::vector<int> m_joint_idx_map;

    // all descendant nodes
    std::vector<std::shared_ptr<RigidBody2<TDataType>>> m_all_childs_nodes;

    // all descendant nodes' pointers and their parent ids
    std::vector<std::pair<int, std::shared_ptr<RigidBody2<TDataType>>>> m_all_childs_node_pairs;

    // system state
    // includes: external force and motion states
    std::shared_ptr<SystemState> m_state;
};

#ifdef PRECISION_FLOAT
template class RigidBodyRoot<DataType3f>;
typedef std::shared_ptr<RigidBodyRoot<DataType3f>> RigidBodyRoot_ptr;
#else
template class RigidBodyRoot<DataType3d>;
typedef std::shared_ptr<RigidBodyRoot<DataType3d>> RigidBodyRoot_ptr;
#endif
}  // namespace PhysIKA