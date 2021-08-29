#pragma once

#include "Framework/Framework/Module.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Dynamics/RigidBody/ForwardDynamicsSolver.h"
#include "Dynamics/RigidBody/RigidBodyRoot.h"

#include <memory>

namespace PhysIKA {
struct State
{
public:
    State()
    {
        //build();
    }

    virtual State operator*(float t) const {}
    virtual State operator+(const State& state) const {}
    //void build();

    static State dydt(const State& s0) {}

    //int getSize() const { return m_v.size(); }
    //const Vectornd<float>& getDq(unsigned int i) const { return m_dq[i]; }
    //const Vectornd<float>& getR(unsigned int i) const { return m_r[i]; }
    //const Quaternion<float>& getQuaternion(unsigned int i) const { return m_qua[i]; }
    //const Vectornd<float>& getV(unsigned int i) const { return m_v[i]; }

    //const MatrixMN<float>& getX(unsigned int i) const { return m_X[i]; }
    //const Vector3f& getGlobalR(unsigned int i) const { return m_global_r[i]; }
    //const Quaternion<float>& getGlobalQ(unsigned int i) const { return m_global_q[i]; }

private:
public:
    //RigidBodyRoot<DataType3f>* m_root = 0;

    // --------

    //std::vector<Vectornd<float>> m_dq;

    //std::vector<Vectornd<float>> m_r;
    //std::vector<Quaternion<float>> m_qua;

    //std::vector<Vectornd<float>> m_v;

    // -----
    //std::vector<MatrixMN<float>> m_X;
    //std::vector<Vector3f> m_global_r;
    //std::vector<Quaternion<float>> m_global_q;
};

}  // namespace PhysIKA