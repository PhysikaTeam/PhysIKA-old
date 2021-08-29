#pragma once

#include "Framework/Framework/Module.h"
#include "Core/Matrix/matrix_mxn.h"
#include "Core/Vector/vector_nd.h"
#include "Framework/Framework/Node.h"

#include <memory>

namespace PhysIKA {
//struct RigidState//:public State
//{
//public:
//    RigidState(Node* root = 0) :m_root(root)
//    {
//        build();
//    }

//    void setRoot(Node * root) { m_root = root; }

//    RigidState operator*(float t)const;
//    RigidState operator+(const RigidState& state) const;
//    void build();

//    static RigidState dydt(const RigidState& s0);

//    int getSize() const { return m_v.size(); }
//    const Vectornd<float>& getDq(unsigned int i) const { return m_dq[i]; }
//    const Vectornd<float>& getR(unsigned int i) const { return m_r[i]; }
//    const Quaternion<float>& getQuaternion(unsigned int i) const { return m_qua[i]; }
//    const Vectornd<float>& getV(unsigned int i) const { return m_v[i]; }

//    const MatrixMN<float>& getX(unsigned int i) const { return m_X[i]; }
//    const Vector3f& getGlobalR(unsigned int i) const { return m_global_r[i]; }
//    const Quaternion<float>& getGlobalQ(unsigned int i) const { return m_global_q[i]; }

//private:
//    template<typename T>
//    std::vector<T> _vecMul(const std::vector<T>& v, float t)const
//    {
//        std::vector<T> tmp_v(v.size());
//        for (int i = 0; i < v.size(); ++i)
//        {
//            tmp_v[i] = v[i] * t;
//        }
//        return tmp_v;
//    }

//    template<typename T>
//    std::vector<T> _vecAdd(const std::vector<T>& v1, const std::vector<T>& v2)const
//    {
//        std::vector<T> tmp_v(v1.size());
//        for (int i = 0; i < v1.size(); ++i)
//        {
//            tmp_v[i] = v1[i] + v2[i];
//        }
//        return tmp_v;
//    }

//    std::vector<Vectornd<float>> _vecMul(const std::vector<Vectornd<float>>& v, float t)const
//    {
//        std::vector<Vectornd<float>> tmp_v(v.size());
//        for (int i = 0; i < v.size(); ++i)
//        {
//            tmp_v[i] = v[i] * t;
//        }
//        return tmp_v;
//    }

//    std::vector<Quaternion<float>> _vecMul(const std::vector<Quaternion<float>>& v, float t)const
//    {
//        std::vector<Quaternion<float>> tmp_v(v.size());
//        for (int i = 0; i < v.size(); ++i)
//        {
//            tmp_v[i] = v[i] * t;
//        }
//        return tmp_v;
//    }

//    std::vector<Vectornd<float>> _vecAdd(const std::vector<Vectornd<float>> & v1, const std::vector<Vectornd<float>>& v2)const
//    {
//        std::vector<Vectornd<float>> tmp_v(v1.size());
//        for (int i = 0; i < v1.size(); ++i)
//        {
//            tmp_v[i] = v1[i] + v2[i];
//        }
//        return tmp_v;
//    }

//    std::vector<Quaternion<float>> _vecAdd(const std::vector<Quaternion<float>> & v1, const std::vector<Quaternion<float>>& v2)const
//    {
//        std::vector<Quaternion<float>> tmp_v(v1.size());
//        for (int i = 0; i < v1.size(); ++i)
//        {
//            tmp_v[i] = (v1[i] + v2[i]);
//        }
//        return tmp_v;
//    }

//public:
//    //RigidBodyRoot<DataType3f>* m_root = 0;
//    Node* m_root = 0;

//    // -------- x
//    std::vector<Vectornd<float>> m_r;
//    std::vector<Quaternion<float>> m_qua;

//    // -------- v
//    std::vector<Vectornd<float>> m_v;
//    std::vector<Vectornd<float>> m_dq;

//    // -------- other information
//    std::vector<MatrixMN<float>> m_X;                    // transformation from predecessor to successor
//    std::vector<Vector3f> m_global_r;
//    std::vector<Quaternion<float>> m_global_q;
//};

}  // namespace PhysIKA