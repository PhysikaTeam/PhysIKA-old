#pragma once

#ifndef PHYSIKA_PBDSOLVERNODE_H
#define PHYSIKA_PBDSOLVERNODE_H

#include "Framework/Framework/Node.h"
#include "Dynamics/RigidBody/PBDRigid/PBDSolver.h"

#include <vector>
#include <memory>

namespace PhysIKA {
//template<typename TReal>
//template<typename TDataType>
class PBDSolverNode : public Node
{
    //DECLARE_CLASS_1(PBDSolverNode, TReal)

public:
    PBDSolverNode();
    //void setUseGPU(bool usegpu = true) { m_useGPU = usegpu; }

    virtual bool initialize() override;

    virtual void advance(Real dt) override;

    int addRigid(RigidBody2_ptr prigid);

    bool setBodyDirty()
    {
        m_solver->setBodyDirty();
    }

    int  addPBDJoint(const PBDJoint<double>& pbdjoint, int bodyid0, int bodyid1);
    bool setJointDirty()
    {
        m_solver->setJointDirty();
    }

    std::shared_ptr<PBDSolver> getSolver()
    {
        return m_solver;
    }
    void setSolver(std::shared_ptr<PBDSolver> solver)
    {
        m_solver = solver;
    }

    void needForward(bool needForward)
    {
        m_needForward = needForward;
    }

protected:
    std::shared_ptr<PBDSolver> m_solver;

    bool m_needForward = true;
};

//template class PBDSolverNode<float>;
//template class PBDSolverNode<double>;

}  // namespace PhysIKA

#endif  // PHYSIKA_PBDSOLVERNODE_H