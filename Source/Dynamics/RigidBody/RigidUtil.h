#pragma once

#ifndef RIGID_UTIL_H
#define RIGID_UTIL_H

#include "Core/Matrix/matrix_mxn.h"
#include "Core/Quaternion/quaternion.h"
//#include "JointSpace.h"

#include "JointSpace.h"
#include "SpatialVector.h"

#ifndef RIGID_ABS
#define RIGID_ABS(x) (x) > 0 ? (x) : (-(x))
#endif  //RIGID_ABS

namespace PhysIKA {

namespace RigidUitlCOMM {
COMM_FUNC inline int maxi(int a, int b)
{
    return a > b ? a : b;
}
COMM_FUNC inline int mini(int a, int b)
{
    return a < b ? a : b;
}
COMM_FUNC inline int maxi(int a, int b, int c)
{
    int res = a > b ? a : b;
    return res > c ? res : c;
}
COMM_FUNC inline int mini(int a, int b, int c)
{
    int res = a < b ? a : b;
    return res < c ? res : c;
}

}  // namespace RigidUitlCOMM

//namespace RigidUtil
class RigidUtil
{
public:
    template <typename MAT>
    static void SwapRow(MAT& A, int i, int j)
    {
        int n = A.cols();
        for (int k = 0; k < n; ++k)
        {
            double tmp = A(i, k);
            A(i, k)    = A(j, k);
            A(j, k)    = tmp;
        }
    }

    template <typename T>
    static void IxS(const MatrixMN<T>& inertia, const JointSpaceBase<T>& S, SpatialVector<T>* res)
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

    template <typename T>
    static void setStxS(const SpatialVector<T>* S1, int dof1, const SpatialVector<T>* S2, int dof2, MatrixMN<T>& H, int idxi, int idxj)
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
    static void setStxS(const SpatialVector<T>* S1, int dof1, const SpatialVector<T>& S2, Vectornd<T>& V, int idx)
    {
        for (int i = 0; i < dof1; ++i)
        {
            V[idx + i] = S1[i] * S2;
        }
    }

    template <typename T>
    static void setStxS(const SpatialVector<T>* S1, int dof1, const SpatialVector<T>& S2, T* V)
    {
        for (int i = 0; i < dof1; ++i)
        {
            V[i] = S1[i] * S2;
        }
    }

    template <typename T>
    static SpatialVector<T> setSxq(const SpatialVector<T>* S1, T* V, int dof1)
    {
        SpatialVector<T> res;
        for (int i = 0; i < dof1; ++i)
        {
            res += S1[i] * V[i];
        }
        return res;
    }

    template <typename T>
    static const void vecSub(const T* v1, const T* v2, int dof, T* res)
    {
        for (int i = 0; i < dof; ++i)
        {
            res[i] = v1[i] - v2[i];
        }
    }
    template <typename T>
    static const void vecAdd(const T* v1, const T* v2, int dof, T* res)
    {
        for (int i = 0; i < dof; ++i)
        {
            res[i] = v1[i] + v2[i];
        }
    }

    template <typename T>
    static const MatrixMN<T> inverse(const MatrixMN<T>& m, int dim)
    {
        T eps = 1e-6;

        MatrixMN<T> res(dim, dim);
        MatrixMN<T> cop(dim, dim);

        /// initialze matrix
        for (int i = 0; i < dim; i++)
        {
            for (int j = 0; j < dim; j++)
            {
                res(i, j) = i == j ? 1 : 0;
                cop(i, j) = m(i, j);
            }
        }

        for (int i = 0; i < dim; i++)
        {
            /// fine main-element
            //int idxi = 0, idxj = 0;
            {
                int idx       = i;
                T   max_value = RIGID_ABS(cop(i, i));
                for (int j = i + 1; j < dim; j++)
                {
                    if (max_value < cop(j, i))
                    {
                        max_value = cop(j, i);
                        idx       = j;
                    }
                    else if (max_value < -cop(j, i))
                    {
                        max_value = -cop(j, i);
                        idx       = j;
                    }
                }

                if (max_value < eps)
                {
                    return res;
                }

                if (idx != i)
                {
                    SwapRow(res, i, idx);
                    SwapRow(cop, i, idx);
                }
            }

            //double ep = mat[i, i];

            T ep = cop(i, i);
            for (int j = 0; j < dim; ++j)
            {
                if (j >= i)
                {
                    cop(i, j) /= ep;
                }
                res(i, j) /= ep;
            }

            for (int j = 0; j < dim; j++)
            {
                if (j != i)
                {
                    T mji = -cop(j, i);
                    for (int k = 0; k < dim; k++)
                    {
                        if (k >= i)
                        {
                            cop(j, k) += mji * cop(i, k);
                        }
                        res(j, k) += mji * res(i, k);
                    }
                }
            }
        }

        return res;
    }

    template <typename T>
    static void setMul(const MatrixMN<T>& m, int nx, int ny, const T* v, T* res)
    {
        for (int i = 0; i < nx; ++i)
        {
            res[i] = 0;
            for (int j = 0; j < ny; ++j)
            {
                res[i] += m(i, j) * v[j];
            }
        }
    }

    template <typename TMAT, typename TVEC>
    static const TVEC MatMulVec(const TMAT& m, const TVEC& v)
    {
        int  nr = m.row();
        int  nc = m.col();
        TVEC res(nr);
        for (int i = 0; i < nr; ++i)
        {
            res[i] = 0;
            for (int j = 0; j < nc; ++j)
            {
                res[i] += m(i, j) * v[j];
            }
        }
        return res;
    }

    //template<typename T>
    //static void IxSj(const Inertia<T>& inertia, const JointSpaceBase<T>& Sj, SpatialVector<T>* res)
    //{
    //    ///???
    //}

    template <typename MAT, typename VEC>
    static bool LinearSolve(const MAT& A, const VEC& b, VEC& x)
    {
        //if (A.cols() != b.size())
        //{
        //    return false;
        //}

        int nx = A.rows();
        int ny = A.cols();
        //if (b.isZero())
        //{
        //    x.resize(m_nx);
        //    x.setZeros();
        //    return true;
        //}

        // ---- Gaussian -----
        VEC tmp_b = b;
        MAT tmp_A = A;
        //VEC tmp_x;

        for (int i = 0; i < nx; ++i)
        {
            // find the largest element
            int    idx_main  = i;
            double max_value = std::abs(tmp_A(i, i));
            for (int j = i + 1; j < nx; ++j)
            {
                if (std::abs(tmp_A(j, i)) > max_value)
                {
                    max_value = std::abs(tmp_A(j, i));
                    idx_main  = j;
                }
            }

            if (idx_main != i)
            {
                // Swap rows
                RigidUtil::SwapRow(tmp_A, i, idx_main);

                // swap value
                //tmp_b.swap(i, idx_main);
                double tmp_value = tmp_b[i];
                tmp_b[i]         = tmp_b[idx_main];
                tmp_b[idx_main]  = tmp_value;
            }
            if (tmp_A(i, i) == 0)
            {
                return false;
            }

            for (int j = i + 1; j < nx; ++j)
            {

                double main_ele = tmp_A(j, i) / tmp_A(i, i);

                tmp_b[j] = tmp_b[j] - tmp_b[i] * main_ele;
                for (int k = i; k < nx; ++k)
                {
                    tmp_A(j, k) = tmp_A(j, k) - tmp_A(i, k) * main_ele;
                }
            }
        }

        x.resize(nx);
        for (int i = nx - 1; i >= 0; --i)
        {
            double sum = 0;
            for (int j = i + 1; j < nx; ++j)
            {
                sum = sum + tmp_A(i, j) * x[j];
            }
            x[i] = (tmp_b[i] - sum) / tmp_A(i, i);
        }

        return true;

        //    //return true;
        //    //int n = this->m_nx;
        //    x.resize(nx);

        //    GeneralMatrix<T>& a = (*this);
        //    GeneralMatrix<T> U(nx, nx);
        //    GeneralVector<T> y(nx);
        //    GeneralVector<T> z(nx);
        //    GeneralMatrix<T> L(nx, nx);
        //    GeneralVector<T> D(nx);

        //    for (int i = 0; i < nx; i++)//用LU先算出L U
        //    {
        //        for (int j = 0; j < nx; j++)
        //        {
        //            U[i][j] = 0;    //暂时全部赋值为0
        //            if (i == j)
        //            {
        //                L[i][j] = 1;//对角线赋值为1
        //            }
        //            else
        //            {
        //                L[i][j] = 0;//其他暂时赋值为0
        //            }
        //        }
        //    }

        //    for (int k = 0; k < nx; k++)//计算u和l矩阵的值
        //    {
        //        for (int j = k; j < nx; j++)
        //        {
        //            U[k][j] = a[k][j];        //第一行
        //            for (int r = 0; r < k; r++)//接下来由L的前一列算u的下一行
        //            {
        //                U[k][j] = U[k][j] - (L[k][r]) * (U[r][j]);
        //            }
        //        }
        //        for (int i = k + 1; i < nx; i++)//计算L的列
        //        {
        //            L[i][k] = a[i][k];
        //            for (int r = 0; r < k; r++)
        //            {
        //                L[i][k] = L[i][k] - (L[i][r]) * (U[r][k]);
        //            }
        //            L[i][k] = L[i][k] / (U[k][k]);
        //        }
        //    }

        //    for (int i = 0; i < nx; i++)//把D赋值
        //    {
        //        D[i] = U[i][i];
        //    }

        //    for (int i = 0; i < nx; i++)//由Lz=b算z
        //    {
        //        z[i] = b[i];
        //        for (int j = 0; j < i; j++)
        //        {
        //            z[i] = z[i] - L[i][j] * z[j];

        //        }
        //    }

        //    for (int i = 0; i < nx; i++)//算y
        //    {
        //        y[i] = z[i] / D[i];
        //    }

        //    GeneralMatrix<T> temp(nx, nx);
        //    for (int i = 0; i < nx; i++)//这里实现对L的转置
        //    {
        //        for (int j = 0; j < nx; j++)
        //        {
        //            temp[i][j] = L[j][i];
        //        }

        //    }
        //    for (int i = 0; i < nx; i++)
        //    {
        //        for (int j = 0; j < nx; j++)
        //        {
        //            L[i][j] = temp[i][j];
        //        }

        //    }

        //    for (int i = nx - 1; i >= 0; i--)//最后算x
        //    {
        //        x[i] = y[i];
        //        for (int j = i + 1; j < nx; j++)
        //        {
        //            x[i] = x[i] - L[i][j] * x[j];
        //        }
        //    }

        //    return true;
        //}

        //static void getTransformationM(MatrixMN<float>& m, const Quaternion<float>& q, const Vectornd<float>& v)
        //{
        //    m.resize(6, 6);
        //    m.setZeros();

        //    MatrixMN<float> rot = RigidUtil::toMatrixMN(q.getConjugate().normalize().get3x3Matrix());
        //    MatrixMN<float> ru;
        //    ru.setCrossMatrix(v);
        //    ru = (rot*ru).negative();
        //
        //    m.setSubMatrix(rot, 0, 0);
        //    m.setSubMatrix(rot, 3, 3);
        //    m.setSubMatrix(ru, 3, 0);

        //}

        //static MatrixMN<float> inverseTransformation(const MatrixMN<float>& tran)
        //{
        //    MatrixMN<float> res(6, 6);
        //    for (int i = 0; i < 3; ++i)
        //    {
        //        for (int j = 0; j < 3; ++j)
        //        {
        //            res[i][j] = tran[j][i];
        //            res[i + 3][j + 3] = tran[j + 3][i + 3];
        //            res[i + 3][j] = tran[j + 3][i];
        //        }
        //    }
        //    return res;
        //}

        //static MatrixMN<float> transformationM2F(const MatrixMN<float>& m)
        //{
        //    MatrixMN<float> res(6,6);

        //    for (int i = 0; i < 3; ++i)
        //    {
        //        for (int j = 0; j < 3; ++j)
        //        {
        //            res[i][j] = m[i][j];
        //            res[i + 3][j + 3] = m[i][j];
        //            res[i][j + 3] = m[i + 3][j];
        //            res[i + 3][j] = 0;
        //        }
        //    }

        //    return res;
        //}

        //static MatrixMN<float> transformationF2M(const MatrixMN<float>& m)
        //{
        //    MatrixMN<float> res(6, 6);

        //    //m.out();

        //    for (int i = 0; i < 3; ++i)
        //    {
        //        for (int j = 0; j < 3; ++j)
        //        {
        //            res[i][j] = m[i][j];
        //            res[i + 3][j + 3] = m[i][j];
        //            res[i + 3][j] = m[i][j + 3];
        //            res[i][j + 3] = 0;
        //        }
        //    }

        //    return res;
        //}

        //static void getTransformationF(MatrixMN<float>& m, const Quaternion<float>& q, const Vectornd<float>& v)
        //{
        //    m.resize(6, 6);
        //    m.setZeros();

        //    MatrixMN<float> rot = RigidUtil::toMatrixMN(q.getConjugate().get3x3Matrix());
        //    MatrixMN<float> ru;
        //    ru.setCrossMatrix(v);
        //    ru = (rot*ru).negative();

        //    m.setSubMatrix(rot, 0, 0);
        //    m.setSubMatrix(rot, 3, 3);
        //    m.setSubMatrix(ru, 0, 3);

        //}

        //static MatrixMN<float> toMatrixMN(const SquareMatrix<float, 3>& m)
        //{
        //    MatrixMN<float> res(3, 3);
        //    for (int i = 0; i < 3; ++i)
        //    {
        //        for (int j = 0; j < 3; ++j)
        //        {
        //            res[i][j] = m(i,j);
        //        }
        //    }
        //    return res;
        //}

        //static void calculateOrthogonalSpace(const MatrixMN<float>& S, MatrixMN<float>& T)
        //{
        //    int D = S.rows();
        //    int dof = S.cols();

        //    T.resize(D, D - dof);

        //    int t_dim = 0;

        //    for (int i = 0; i < D; ++i)
        //    {
        //        Vectornd<float> e(D);
        //        e(i) = 1;

        //        for (int j = 0; j < dof; ++j)
        //        {
        //            Vectornd<float> cur_col = S.getCol(j);
        //            cur_col.normalize();

        //            e = e - cur_col *  (e*cur_col);
        //        }

        //        if (!e.isZero())
        //        {
        //            e.normalize();
        //            T.setCol(e, t_dim++);
        //        }

        //        if (t_dim >= (D - dof))
        //        {
        //            return;
        //        }
        //    }

        //}

        //static MatrixMN<float> crm(const Vectornd<float>& v6)
        //{
        //    MatrixMN<float> res(6, 6);

        //    res[0][0] = 0;        res[0][1] = -v6[2];    res[0][2] = v6[1];
        //    res[1][0] = v6[2];    res[1][1] = 0;        res[1][2] = -v6[0];
        //    res[2][0] = -v6[1];    res[2][1] = v6[0];    res[2][2] = 0;

        //    res[3][0] = 0;        res[3][1] = -v6[5];    res[3][2] = v6[4];
        //    res[4][0] = v6[5];    res[4][1] = 0;        res[4][2] = -v6[3];
        //    res[5][0] = -v6[4];    res[5][1] = v6[3];    res[5][2] = 0;

        //    res[3][3] = 0;        res[3][4] = -v6[2];    res[3][5] = v6[1];
        //    res[4][3] = v6[2];    res[4][4] = 0;        res[4][5] = -v6[0];
        //    res[5][3] = -v6[1];    res[5][4] = v6[0];    res[5][5] = 0;

        //    return res;
        //}

        //static MatrixMN<float> crf(const Vectornd<float>& v6)
        //{
        //    return crm(v6).transpose().negative();
        //}

        //static Vector3f toVector3f(const Vectornd<float>& v)
        //{
        //    return Vector3f(v[0], v[1], v[2]);
        //}

        //static Vectornd<float> toVectornd(const Vector3f& v)
        //{
        //    Vectornd<float> res(3);
        //    res[0] = v[0];    res[1] = v[1];    res[2] = v[2];
        //    return res;
        //}

        //static Vectornd<float> rotateOnly(const MatrixMN<float>& X, const Vectornd<float>& f)
        //{
        //    Vectornd<float> res(6);
        //    for (int i = 0; i < 3; ++i)
        //    {
        //        for (int j = 0; j < 3; ++j)
        //        {
        //            res[i] = res[i] + f[j] * X[i][j];
        //            res[i + 3] = res[i + 3] + f[j + 3] * X[i][j];
        //        }
        //    }

        //    return res;
        //}

        //static Vectornd<float> toVectorSubspace(const MatrixMN<float>& S, const Vectornd<float>& v)
        //{
        //    //MatrixMN<float> SST = S * S.transpose();
        //    //Vectornd<float> vs = SST * v;

        //    //for (int i = 0; i < v.size(); ++i)
        //    //{
        //    //    vs[i] = vs[i] / SST[i][i];
        //    //}

        //    //return vs;

        //    Vectornd<float> q(S.cols());
        //    MatrixMN<float> S_tran = S.transpose();

        //    for (int i = 0; i < S.cols(); ++i)
        //    {
        //        float nor = S_tran[i].norm();
        //        if (nor != 0)
        //        {
        //            q[i] = S_tran[i] * v / (nor * nor);
        //        }
        //    }

        //    Vectornd<float> res = S*q;

        //    return res;
    }

    static Vector3f calculateCubeLocalInertia(float mass, const Vector3f& geoSize)
    {
        Vector3f iner;
        iner[0] = mass / 12.0f * (geoSize[1] * geoSize[1] + geoSize[2] * geoSize[2]);
        iner[1] = mass / 12.0f * (geoSize[0] * geoSize[0] + geoSize[2] * geoSize[2]);
        iner[2] = mass / 12.0f * (geoSize[0] * geoSize[0] + geoSize[1] * geoSize[1]);
        return iner;
    }

    static Vector3f calculateCylinderLocalInertia(float mass, float radius, float height, int axis = 0)
    {
        Vector3f iner;
        for (int i = 0; i < 3; ++i)
        {
            if (i == axis)
                iner[i] = 0.5f * mass * radius;
            else
                iner[i] = 1.0f / 12.0f * mass * (3.0f * radius * radius + height * height);
        }
        return iner;
    }

    static Vector3f calculateSphereLocalInertia(float mass, float radius)
    {
        float inertia = 2.0 / 5.0 * mass * radius * radius;
        return Vector3f(inertia, inertia, inertia);
    }
};
}  // namespace PhysIKA

#endif  // RIGID_UTIL_H