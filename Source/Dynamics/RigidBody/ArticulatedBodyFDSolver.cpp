#include "ArticulatedBodyFDSolver.h"
#include "RigidBodyRoot.h"
#include "SpatialVector.h"
#include <vector>
#include "RigidUtil.h"

namespace PhysIKA {

template <typename T>
const SpatialVector<T> _Ixs(const MatrixMN<T>& m, const SpatialVector<T>& v)
{
    SpatialVector<T> res;
    for (int i = 0; i < 6; ++i)
    {
        res[i] = 0;
        for (int j = 0; j < 6; ++j)
        {
            res[i] += m(i, j) * v[j];
        }
    }
    return res;
}

/**
    * @brief Calculate matrix transformation. U * D * U^T
    * @param U
    * @param dof dof of U
    * @param D
    * @return U * D * U^T
    */
template <typename T>
const MatrixMN<T> UxDxUT(const SpatialVector<T>* U, int dof, const MatrixMN<T>& D)
{
    MatrixMN<float> res(6, 6);

    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < dof; ++j)
        {
            res(i, j) = 0;
            for (int k = 0; k < dof; ++k)
            {
                res(i, j) += U[k][i] * D(k, j);
            }
        }
    }

    T tmp[6];
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            tmp[j] = 0;
            for (int k = 0; k < dof; ++k)
            {
                tmp[j] += res(i, k) * U[k][j];
            }
        }
        res(i, 0) = tmp[0];
        res(i, 1) = tmp[1];
        res(i, 2) = tmp[2];
        res(i, 3) = tmp[3];
        res(i, 4) = tmp[4];
        res(i, 5) = tmp[5];
    }

    return res;
}

/**
    * @brief Calculate matrix transformation. U * D * ui
    * @param U
    * @param D
    * @param ui
    * @param dof dof of U, D and ui
    * @return U * D * U^T
    */
template <typename T>
const SpatialVector<T> UxDxui(const SpatialVector<T>* U, const MatrixMN<T>& D, const T* ui, int dof)
{
    SpatialVector<T> res;
    T                tmpv[6];
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < dof; ++j)
        {
            tmpv[j] = 0;
            for (int k = 0; k < dof; ++k)
            {
                tmpv[j] += U[k][i] * D(k, j);
            }
        }

        res[i] = 0;
        for (int j = 0; j < dof; ++j)
        {
            res[i] += tmpv[j] * ui[j];
        }
    }

    return res;
}

bool ArticulatedBodyFDSolver::solve(const SystemState& s_system, const SystemMotionState& s, Vectornd<float>& ddq)
{
    RigidBodyRoot<DataType3f>* root    = static_cast<RigidBodyRoot<DataType3f>*>(s.m_root);
    const std::vector<int>&    idx_map = root->getJointIdxMap();

    auto all_bodies = root->getAllParentidNodePair();
    int  n_rigid    = all_bodies.size();

    m_IA.resize(n_rigid);
    m_pA.resize(n_rigid);
    m_U.resize(n_rigid);
    m_D_inv.resize(n_rigid);
    m_ui.resize(root->getJointDof());
    m_a.resize(n_rigid);

    ddq.resize(root->getJointDof());

    // *** Pass 1 :  *******************
    // Calculate rigid global velocity and bias forces.

    std::vector<SpatialVector<float>> vi(n_rigid);
    std::vector<SpatialVector<float>> ci(n_rigid);
    //std::vector<Transform3d<float>> X0(n_rigid);
    for (int i = 0; i < n_rigid; ++i)
    {
        RigidBody2_ptr cur_node     = all_bodies[i].second;
        Joint*         parent_joint = cur_node->getParentJoint();
        int            parent_id    = all_bodies[i].first;
        int            cur_id       = cur_node->getId();

        if (parent_id >= 0)
        {
            vi[cur_id] = s.m_X[cur_id].transformM(vi[parent_id]) + s.m_v[cur_id];

            //X0[cur_id] = X0[parent_id] *
        }
        else
        {
            vi[cur_id] = s.m_v[cur_id];
        }

        ci[cur_id] = vi[cur_id].crossM(s.m_v[cur_id]);

        cur_node->getI().getTensor(m_IA[cur_id]);

        Transform3d<float> X0(Vector3f(0, 0, 0), s.globalRotation[cur_id].getConjugate());
        m_pA[cur_id] = vi[cur_id].crossF(cur_node->getI() * vi[cur_id]);
        m_pA[cur_id] -= X0.transformF(s_system.m_externalForce[cur_id]);
    }

    // *** Pass 2 :  *******************
    // Calculate articulated body inertia.

    MatrixMN<float> D(6, 6);
    //Vectornd<float> ui(6);
    MatrixMN<float>      Ia(6, 6);
    SpatialVector<float> pa;

    for (int i = n_rigid - 1; i >= 0; --i)
    {
        RigidBody2_ptr cur_node     = all_bodies[i].second;
        Joint*         parent_joint = cur_node->getParentJoint();
        int            parent_id    = all_bodies[i].first;
        int            cur_id       = cur_node->getId();

        int cur_dof = parent_joint->getJointDOF();
        //SpatialVector<float> U[6];
        SpatialVector<float>* U = (this->m_U[cur_id]).getBases();

        /// Ui and Di
        //if (!m_isValid)
        if (cur_dof > 0)
        {
            // U = IA * S
            RigidUtil::IxS(m_IA[cur_id], parent_joint->getJointSpace(), U);

            // D = S^T * U = S^T * IA * S
            RigidUtil::setStxS(parent_joint->getJointSpace().getBases(), cur_dof, U, cur_dof, D, 0, 0);

            // ui = t - S^T * pA
            RigidUtil::setStxS(parent_joint->getJointSpace().getBases(), cur_dof, m_pA[cur_id], m_ui, idx_map[cur_id]);
            RigidUtil::vecSub(&(s_system.m_activeForce[idx_map[cur_id]]), &(m_ui[idx_map[cur_id]]), cur_dof, &(m_ui[idx_map[cur_id]]));

            // D^-1 = (S^T *IA * S)^-1
            this->m_D_inv[cur_id] = RigidUtil::inverse(D, cur_dof);
        }

        if (parent_id >= 0)
        {
            MatrixMN<float>& D_inv = this->m_D_inv[cur_id];

            if (cur_dof > 0)
            {
                // Ia = IA - IA * S * (S^T * IA *S)^-1 * S^T * IA
                // => Ia = IA - U * D^-1 * U^T
                Ia = m_IA[cur_id] - UxDxUT(U, cur_dof, D_inv);

                // pa = pA + Ia * c + IA * S * (S^T * IA * S)^-1 * (t - S^T * pA)
                // => pa = pA + Ia * c + U * D^-1 * ui
                pa = m_pA[cur_id] + UxDxui(U, D_inv, &(m_ui[idx_map[cur_id]]), cur_dof) + _Ixs(Ia, ci[cur_id]);

                m_IA[parent_id] += s.m_X[cur_id].inverseTransform().transformI(Ia);
                m_pA[parent_id] += s.m_X[cur_id].inverseTransform().transformF(pa);
            }
            else
            {
                m_IA[parent_id] += s.m_X[cur_id].inverseTransform().transformI(m_IA[cur_id]);
                m_pA[parent_id] += s.m_X[cur_id].inverseTransform().transformF(m_pA[cur_id]);
            }
        }
    }

    // *** Pass 3 :  *******************
    // Calculate accelerations.

    for (int i = 0; i < n_rigid; ++i)
    {
        RigidBody2_ptr cur_node     = all_bodies[i].second;
        Joint*         parent_joint = cur_node->getParentJoint();
        int            parent_id    = all_bodies[i].first;
        int            cur_id       = cur_node->getId();
        int            cur_dof      = parent_joint->getJointDOF();
        int            start_idx    = idx_map[cur_id];

        SpatialVector<float> a_p;
        if (parent_id < 0)
        {
            Transform3d<float> toNode(Vector3f(), s.globalRotation[i].getConjugate());
            a_p = SpatialVector<float>(Vector3f(), -(root->getGravity()));
            a_p = toNode.transformM(a_p) + ci[cur_id];
        }
        else
        {
            // ap = a[parent] + c
            a_p = s.m_X[cur_id].transformM(m_a[parent_id]) + ci[cur_id];
        }

        if (cur_dof > 0)
        {
            float a_[6];

            // a_ = S^T * IA * (a[parent] + c)
            m_U[cur_id].transposeMul(a_p, a_);

            // ui - a_ = t - S^T * pA -  S^T * IA * (a[parent] + c)
            RigidUtil::vecSub(&(m_ui[start_idx]), a_, cur_dof, a_);

            // ddq = D^-1 * (ui - a_)
            // ddq = (S^T * IA * S)^-1 * (t - S^T * pA -  S^T * IA * (a[parent] + c))
            RigidUtil::setMul(m_D_inv[cur_id], cur_dof, cur_dof, a_, &(ddq[start_idx]));

            m_a[cur_id] = a_p + parent_joint->getJointSpace().mul(&(ddq[start_idx]));
        }
        else
        {
            m_a[cur_id] = a_p;
        }
    }

    return true;
}

void ArticulatedBodyFDSolver::init()
{

    //RigidBodyRoot<DataType3f>* root = static_cast<RigidBodyRoot<DataType3f>*>(s.m_root);
    //const std::vector<int>& idx_map = root->getJointIdxMap();

    //auto all_bodies = root->getAllParentidNodePair();
    //int n_rigid = all_bodies.size();

    ////if()
    //m_IA.resize(n_rigid);
    //m_pA.resize(n_rigid);
    //m_U.resize(n_rigid);
    //m_D_inv.resize(n_rigid);
    //m_ui.resize(root->getJointDof());
}

}  // namespace PhysIKA