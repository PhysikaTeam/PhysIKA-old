//
//#include "Dynamics/RigidBody/RigidUtil.h"
//
//namespace PhysIKA
//{
//    namespace RigidUtil
//    {
//        void getTransformation(MatrixMN<float>& m, const Quaternion<float>& q, const Vectornd<float>& v)
//        {
//            m.resize(6, 6);
//            m.setZeros();
//
//            // rotation part
//            SquareMatrix<float, 3> rot = q.get3x3Matrix();
//            for (int i = 0; i < 3; ++i)
//            {
//                for (int j = 0; j < 3; ++j)
//                {
//                    m(i, j) = rot(i, j);
//                    m(i + 3, j + 3) = rot(i, j);
//                }
//            }
//
//            // translation part
//            m(3, 0) = 0;        m(3, 1) = v(2);        m(3, 2) = -v(1);
//            m(4, 0) = -v(2);    m(4, 1) = 0;        m(4, 2) = v(0);
//            m(5, 0) = v(1);        m(5, 1) = -v(1);    m(5, 2) = 0;
//
//        }
//
//
//        void calculateOrthogonalSpace(const MatrixMN<float>& S, MatrixMN<float>& T)
//        {
//            int D = S.rows();
//            int dof = S.cols();
//
//            T.resize(D, D - dof);
//
//            int t_dim = 0;
//
//            for (int i = 0; i < D; ++i)
//            {
//                Vectornd<float> e(D);
//                e(i) = 1;
//
//                for (int j = 0; j < dof; ++j)
//                {
//                    Vectornd<float> cur_col = S.getCol(j);
//                    cur_col.normalize();
//
//                    e = e - cur_col *  (e*cur_col);
//                }
//
//                if (!e.isZero())
//                {
//                    e.normalize();
//                    T.setCol(e, t_dim++);
//                }
//
//                if (t_dim >= (D - dof))
//                {
//                    return;
//                }
//            }
//
//        }
//
//    }
//}