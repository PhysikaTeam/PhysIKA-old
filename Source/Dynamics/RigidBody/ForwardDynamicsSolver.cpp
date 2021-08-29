
#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/RigidUtil.h"
#include "Transform3d.h"

#include "Dynamics/RigidBody/RigidBodyRoot.h"
#include "Framework/Action/Action.h"
#include "Framework/Framework/Base.h"
#include <queue>
#include <memory>
#include <vector>

namespace PhysIKA {
//template<typename T>
//void setStxS(const JointSpaceBase<T>& S1, const SpatialVector<T>* S2, MatrixMN<T>& H, int idxi, int idxj);

//template<typename T>
//void setStxS(const SpatialVector<T>* S1, const JointSpaceBase<T>& S2, MatrixMN<T>& H, int idxi, int idxj);
//void

void _outMatrixMN(const MatrixMN<float>& m)
{
    int nx = m.rows();
    int ny = m.cols();
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            std::cout << m(i, j) << "   ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

void _outVec(const SpatialVector<float>& v)
{
    for (int i = 0; i < 6; ++i)
    {
        std::cout << v[i] << "   ";
    }
    std::cout << std::endl;
}

const SpatialVector<float> _v2p(const MatrixMN<float>& m, const SpatialVector<float>& v)
{
    SpatialVector<float> res;
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

template <typename T>
void setStxS(const SpatialVector<T>* S1, int dof1, const SpatialVector<T>* S2, int dof2, MatrixMN<T>& H, int idxi, int idxj)
{
    for (int i = 0; i < dof1; ++i)
    {
        for (int j = 0; j < dof2; ++j)
        {
            H(idxi + i, idxj + j) = S1[i] * S2[j];
        }
    }
}

template <typename T>
void transformFs(const Transform3d<T>& X, SpatialVector<T>* f, int n)
{
    for (int i = 0; i < n; ++i)
    {
        f[i] = X.transformF(f[i]);
    }
}

template <typename T>
void setSymmetric(MatrixMN<T>& H, int idxi, int idxj, int li, int lj)
{
    for (int i = idxi; i < (idxi + li); ++i)
    {
        for (int j = idxj; j < (idxj + lj); ++j)
        {
            H(j, i) = H(i, j);
        }
    }
}

template <typename T>
void transformI(const Transform3d<T>& trans, const MatrixMN<T>& inertia, MatrixMN<T>& res)
{
    //MatrixMN<T> res(6, 6);
    res.resize(6, 6);

    // I_tmp = X_12f * I_1
    for (int i = 0; i < 6; ++i)
    {
        SpatialVector<T> tmpv(inertia(0, i), inertia(1, i), inertia(2, i), inertia(3, i), inertia(4, i), inertia(5, i));
        //res[i] = this->transformF(tmpv);
        SpatialVector<T> tmpres = this->transformF(tmpv);

        res(i, 0) = tmpres[0];
        res(i, 1) = tmpres[1];
        res(i, 2) = tmpres[2];
        res(i, 3) = tmpres[3];
        res(i, 4) = tmpres[4];
        res(i, 5) = tmpres[5];
    }

    // I_2 = I_tmp * X_21m
    // ==> I_2 = I_2^T = (I_tmp * X_21m)^T = X_12f * I_tmp^T
    for (int i = 0; i < 6; ++i)
    {
        SpatialVector<T> tmpv(res(0, i), res(1, i), res(2, i), res(3, i), res(4, i), res(5, i));
        SpatialVector<T> tmpres = this->transformF(tmpv);

        res(0, i) = tmpres[0];
        res(1, i) = tmpres[1];
        res(2, i) = tmpres[2];
        res(3, i) = tmpres[3];
        res(4, i) = tmpres[4];
        res(5, i) = tmpres[5];
    }
}

template <typename T>
void IxS(const MatrixMN<T>& inertia, const JointSpaceBase<T>& S, SpatialVector<T>* res)
{
    int joint_dof = S.dof();

    for (int i = 0; i < joint_dof; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            res[i][j] = 0;
            for (int k = 0; k < 6; ++k)
            {
                res[i][j] += inertia(j, k) * S(k, i);
            }
        }
    }
}

IMPLEMENT_CLASS(InertiaMatrixFDSolver)

inline InertiaMatrixFDSolver::InertiaMatrixFDSolver()
{
}

void InertiaMatrixFDSolver::buildJointSpaceMotionEquation(const SystemState& s_system, const SystemMotionState& motion_state, MatrixMN<float>& H, Vectornd<float>& C)
{
    //const SystemMotionState& motion_state = *(state.m_motionState);
    //const std::vector<SpatialVector<float>>& external_force = s_system.m_externalForce;

    RigidBodyRoot<DataType3f>*                                           root      = static_cast<RigidBodyRoot<DataType3f>*>(this->getParent());
    std::vector<std::pair<int, std::shared_ptr<RigidBody2<DataType3f>>>> all_node  = root->getAllParentidNodePair();
    const std::vector<int>&                                              idx_map   = root->getJointIdxMap();
    int                                                                  joint_dof = root->getJointDof();

    std::vector<SpatialVector<float>> v(all_node.size());    // acceleration, node frame
    std::vector<SpatialVector<float>> avp(all_node.size());  // velocity, node frame
    std::vector<SpatialVector<float>> fvp(all_node.size());

    std::vector<MatrixMN<float>> IC(all_node.size());

    /// ***  calculate C, using inverse dynamics *************
    /// This part consists of 2 passes.

    /// The first pass
    for (int i = 0; i < all_node.size(); ++i)
    {
        int                                     parent_id = all_node[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_node[i].second;
        Joint*                                  cur_joint = cur_node->getParentJoint();

        /// transformation from parent node to current node.
        const Transform3d<float>& Xup = motion_state.m_X[i];

        /// ****** debug
        Vector3f          Xup_translation = Xup.getTranslation();
        Quaternion<float> Xup_rotation    = Xup.getRotation();

        SpatialVector<float> vJ;
        if (cur_joint->getJointDOF() > 0)
        {
            /// Joint space matrix, expressed in successor frame.
            const JointSpaceBase<float>& Si = cur_joint->getJointSpace();

            /// Relative velocity, expressed in successor frame.
            vJ = Si.mul(&(motion_state.generalVelocity[idx_map[i]]));
        }
        else
        {
            vJ = SpatialVector<float>(0, 0, 0, 0, 0, 0);
        }

        if (parent_id < 0)
        {
            v[i] = (vJ);

            Transform3d<float>   toNode(Vector3f(), motion_state.globalRotation[i].getConjugate());
            SpatialVector<float> a0(Vector3f(), -(root->getGravity()));
            a0 = toNode.transformM(a0);

            avp[i] = a0;  //SpatialVector<float>();
        }
        else
        {
            /// vi = v_parent + vJ;
            /// note: vi is expressed in current node frame.
            v[i] = (Xup.transformM(v[parent_id]) + vJ);

            /// ai = a_parent + v x vJ;
            /// note: it is expressed in cerrent node frame.
            avp[i] = Xup.transformM(avp[parent_id]) + v[i].crossM(vJ);
        }

        ///
        fvp[i] = cur_node->getI() * avp[i];
        fvp[i] = fvp[i] + v[i].crossF(cur_node->getI() * v[i]);

        /// Transformation from world to node.
        /// Suspect the origin of the external_force is the same as the origin of node frame.
        Transform3d<float> Xo(Vector3f(0, 0, 0), motion_state.globalRotation[i].getConjugate());

        fvp[i] = fvp[i] - Xo.transformF(s_system.m_externalForce[i]);
    }

    /// Calculation of C, the second part.
    C.resize(joint_dof);
    C.setZeros();
    for (int i = all_node.size() - 1; i >= 0; --i)
    {
        int                                     parent_id = all_node[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_node[i].second;
        Joint*                                  cur_joint = cur_node->getParentJoint();

        if (cur_joint->getJointDOF() > 0)
        {
            /// Map the force into joint space vector
            cur_joint->getJointSpace().transposeMul(fvp[i], &(C[idx_map[i]]));

            int cur_dof = cur_joint->getJointDOF();
            for (int j = 0; j < cur_dof; ++j)
            {
                C[idx_map[i] + j] = -C[idx_map[i] + j];
            }
        }

        if (parent_id >= 0)
        {
            /// Transformation from predecessor to successor
            const Transform3d<float> Xup = motion_state.m_X[i];

            fvp[parent_id] = fvp[parent_id] + Xup.inverseTransform().transformF(fvp[i]);
        }

        //IC[i] = cur_node->getI();
        cur_node->getI().getTensor(IC[i]);
    }

    /// ****** Calcualte H, inertia matrix method ***********

    H.resize(joint_dof, joint_dof);
    H.setZeros();
    for (int i = all_node.size() - 1; i >= 0; --i)
    {
        int                                     parent_id = all_node[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_node[i].second;
        Joint*                                  cur_joint = cur_node->getParentJoint();

        //if (cur_joint->getJointDOF() > 0)
        if (parent_id >= 0)
        {
            IC[parent_id] = IC[parent_id] + motion_state.m_X[i].inverseTransform().transformI(IC[i]);
        }
    }

    for (int i = 0; i < all_node.size(); ++i)
    {
        int                                     parent_id = all_node[i].first;
        std::shared_ptr<RigidBody2<DataType3f>> cur_node  = all_node[i].second;
        Joint*                                  cur_joint = cur_node->getParentJoint();

        if (cur_joint->getJointDOF() > 0)
        {
            //const JointSpace<float>& Si = cur_joint->getJointSpace();
            const JointSpaceBase<float>& Si   = cur_joint->getJointSpace();
            int                          dofi = Si.dof();

            SpatialVector<float> F[6];

            // F = Ic_i * S_i
            IxS(IC[i], Si, &(F[0]));

            /// ***** debug
            SpatialVector<float> tmp_f = _v2p(IC[i], Si.getBases()[0]);

            // H_ii = S_i^T * F
            setStxS(Si.getBases(), dofi, F, dofi, H, idx_map[i], idx_map[i]);

            int j = i;
            while (all_node[j].first >= 0)
            {

                // F = Xup[j].transpose() * F;
                transformFs(motion_state.m_X[j].inverseTransform(), F, dofi);

                j                                              = all_node[j].first;
                std::shared_ptr<RigidBody2<DataType3f>> nodej  = all_node[j].second;
                Joint*                                  jointj = nodej->getParentJoint();

                if (jointj->getJointDOF() > 0)
                {
                    setStxS(F, dofi, jointj->getJointSpace().getBases(), jointj->getJointDOF(), H, idx_map[i], idx_map[j]);

                    // H_ji = H_ij^T
                    setSymmetric(H, idx_map[i], idx_map[j], dofi, jointj->getJointDOF());
                }
            }
        }
    }

    //_outMatrixMN(H);
}

bool InertiaMatrixFDSolver::solve(const SystemState& s_system, const SystemMotionState& s, Vectornd<float>& ddq)
{
    // build equation
    this->buildJointSpaceMotionEquation(s_system, s, m_H, m_C);

    // sovlve equation
    bool res = RigidUtil::LinearSolve(m_H, m_C, ddq);

    //Vectornd<float> tmp_C = m_H * ddq;
    //Vectornd<float> tmp_dif = tmp_C - m_C;
    //for (int i = 0; i < tmp_dif.size(); ++i)
    //{
    //    std::cout << tmp_dif[i] << "  ";
    //}
    //std::cout << std::endl;

    return res;
}

}  // namespace PhysIKA