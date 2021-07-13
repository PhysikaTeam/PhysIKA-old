
#include "Dynamics/RigidBody/RigidTimeIntegrationModule.h"
#include "Framework/Framework/Module.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"

#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Framework/Action/Action.h"
#include "Dynamics/RigidBody/RKIntegrator.h"
#include "Core/Utility/CTimer.h"
#include <iostream>
#include <queue>
#include <memory>
#include <vector>
#include <time.h>

#include "SemiImplicitEuler.h"
#include "ArticulatedBodyFDSolver.h"

namespace PhysIKA {

IMPLEMENT_CLASS(RigidTimeIntegrationModule);

RigidTimeIntegrationModule::RigidTimeIntegrationModule()
{
    m_fd_solver = std::make_shared<ArticulatedBodyFDSolver>();
    m_fd_solver->setParent(this->getParent());
}

inline void RigidTimeIntegrationModule::begin()
{
    //m_fd_solver.setParent(this->getParent());

    //if (!m_time_init)
    //{
    //    m_dt = 0.001;
    //    m_last_time = clock() / 1000.0;
    //    m_time_init = true;
    //}
    //else
    //{
    //    double cur_time = clock() / 1000.0;
    //    m_dt = cur_time - m_last_time;
    //    m_last_time = cur_time;
    //}
}

inline bool RigidTimeIntegrationModule::execute()
{

    //return true;

    //m_dt = 1.0 / 60;
    //RigidState s(static_cast<RigidBodyRoot<DataType3f>*>(this->getParent()));
    RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(this->getParent());
    SystemState&               s    = *(static_cast<RigidBodyRoot<DataType3f>*>(this->getParent())->getSystemState());

    RK4Integrator rk4;
    //*(s.m_motionState) = rk4.solve(*(s.m_motionState), dydt, m_dt);
    rk4.solve(*(s.m_motionState), DydtAdapter(this), m_dt);

    //s = s + RigidState::dydt(s) * m_dt;

    //SemiImplicitEulerIntegrator semi;
    //semi.solve(s, m_dt);

    //updateSystemState(s);
    return true;
}

void RigidTimeIntegrationModule::dydt(const SystemMotionState& s0, DSystemMotionState& ds)
{
    RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(s0.m_root);
    this->dydt(*(root->getSystemState()), s0, ds);
}

void RigidTimeIntegrationModule::dydt(const SystemState& sysState, const SystemMotionState& s0, DSystemMotionState& ds)
{
    CTimer timer;
    timer.start();

    RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(s0.m_root);
    //std::shared_ptr<SystemState> system_state = root->getSystemState();

    CTimer timer_solve_FD;
    timer_solve_FD.start();

    /// Use InertiaMatrix forward dynamics solver
    //InertiaMatrixFDSolver fd_solver;
    //ArticulatedBodyFDSolver& fd_solver = m_fd_solver;
    //ArticulatedBodyFDSolver fd_solver;
    //fd_solver.setParent(root);

    /// Solve general acceleration.
    bool solve_ok = m_fd_solver->solve(sysState, s0, ds.m_dq);

    timer_solve_FD.stop();
    //std::cout << "  TIME * solve FD: " << timer_solve_FD.getElapsedTime() << std::endl;

    if (solve_ok)
    {

        CTimer timer_dy;
        timer_dy.start();

        const std::vector<std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>>& all_nodes = root->getAllParentidNodePair();

        /// joint index map
        const std::vector<int>& idx_map = root->getJointIdxMap();

        ds.setRigidNum(all_nodes.size());
        ds.setDof(root->getJointDof());

        for (int i = 0; i < all_nodes.size(); ++i)
        {
            int                                     parent_id = all_nodes[i].first;
            std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_nodes[i].second;
            Joint*                                  cur_joint = cur_node->getParentJoint();

            Vectornd<float>& generalq = ds.getGeneralFreedom();
            int              dof      = cur_joint->getJointDOF();
            for (int di = 0; di < dof; ++di)
            {
                int idx0     = idx_map[i];
                int violateC = cur_joint->violateConstraint(di, s0.getGeneralPosition()[idx0 + di]);
                if (violateC < 0)
                {
                    generalq[idx0 + di] = max(0, s0.generalVelocity[idx0 + di]);
                    ds.m_dq[idx0 + di]  = max(0, ds.m_dq[idx0 + di]);
                }
                else if (violateC > 0)
                {
                    generalq[idx0 + di] = min(0, s0.generalVelocity[idx0 + di]);
                    ds.m_dq[idx0 + di]  = min(0, ds.m_dq[idx0 + di]);
                }
                else
                    generalq[idx0 + di] = s0.generalVelocity[idx0 + di];
            }

            if (dof > 0)
            {
                const Transform3d<float>& Xup    = s0.m_X[i];
                Transform3d<float>        Xupinv = Xup.inverseTransform();

                /// d_vJ
                ds.m_v[i] = cur_joint->getJointSpace().mul(&(ds.m_dq[idx_map[i]]));

                SpatialVector<float> cur_v6 = s0.m_v[i];

                /// d_rel_q
                ds.m_rel_q[i] = s0.m_rel_q[i] * Quaternion<float>(cur_v6[0], cur_v6[1], cur_v6[2], 0) * 0.5;

                /// d_rel_r
                cur_v6 = Xupinv.transformM(cur_v6);
                Vector3f cur_w;
                cur_w[0] = cur_v6[0];
                cur_w[1] = cur_v6[1];
                cur_w[2] = cur_v6[2];
                Vector3f d_r;
                d_r[0]        = cur_v6[3];
                d_r[1]        = cur_v6[4];
                d_r[2]        = cur_v6[5];
                ds.m_rel_r[i] = d_r + cur_w.cross(s0.m_rel_r[i]);  ///< 3dim velocity: v = v0 + w x r;
            }
            else
            {
                /// d_vJ
                ds.m_v[i] = SpatialVector<float>();  // d_vJ, expressed in node frame

                // d_r, d_quaternion
                ds.m_rel_q[i] = Quaternion<float>(0, 0, 0, 0);
                ds.m_rel_r[i] = Vector3f();
            }
        }

        timer_dy.stop();
        //std::cout << "  TIME * ddq to dr&dQ: " << timer_dy.getElapsedTime() << std::endl;

        timer.stop();
        //std::cout << "TIME * dydt:  " << timer.getElapsedTime() << std::endl;

        //return ds;
    }
}

void RigidTimeIntegrationModule::setFDSolver(std::shared_ptr<ForwardDynamicsSolver> fd_solver)
{
    this->m_fd_solver = fd_solver;
    this->m_fd_solver->setParent(this->getParent());
}

}  // namespace PhysIKA