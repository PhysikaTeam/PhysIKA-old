/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: some useful interpolation.
 * @version    : 1.0
 */
#pragma once
#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "nanoflann/nanoflann.hpp"
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <ANN/ANN.h>
#include "../Common/eigen_ext.h"
#include <iostream>
#include <vector>
#include <memory>
#include "get_nn.h"

namespace PhysIKA {
template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
    };

    std::vector<Point> pts;

    // Must return the number of data points
    inline size_t kdtree_get_point_count() const
    {
        return pts.size();
    }

    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

template <typename FLOAT, size_t dim_>
Eigen::Vector4i get_noncoplanar_tet(const Eigen::Matrix<FLOAT, dim_, -1>& v, const std::vector<size_t>& neigh_vert_idx, const Eigen::Matrix<FLOAT, dim_, 1>& p);
template <typename T, size_t dim_>
int interp_pts_in_point_cloud(const Eigen::Matrix<T, dim_, -1>& v, const Eigen::Matrix<T, dim_, -1>& pts, Eigen::SparseMatrix<T>& coef);

inline Eigen::MatrixXi hex_2_tet(const Eigen::Matrix<int, 8, -1>& hexs)
{
    Eigen::MatrixXi tets = Eigen::MatrixXi::Zero(4, 5 * hexs.cols());
    for (size_t hex_id = 0; hex_id < hexs.cols(); ++hex_id)
    {
        const Eigen::VectorXi one_hex = hexs.col(hex_id);
        Eigen::Vector4i       tet0{ { one_hex(0), one_hex(1), one_hex(3), one_hex(4) } };
        Eigen::Vector4i       tet1{ { one_hex(1), one_hex(2), one_hex(3), one_hex(6) } };
        Eigen::Vector4i       tet2{ { one_hex(4), one_hex(5), one_hex(6), one_hex(1) } };
        Eigen::Vector4i       tet3{ { one_hex(4), one_hex(6), one_hex(7), one_hex(3) } };
        Eigen::Vector4i       tet4{ { one_hex(1), one_hex(3), one_hex(4), one_hex(6) } };
        tets.col(hex_id * 5)     = tet0;
        tets.col(hex_id * 5 + 1) = tet1;
        tets.col(hex_id * 5 + 2) = tet2;
        tets.col(hex_id * 5 + 3) = tet3;
        tets.col(hex_id * 5 + 4) = tet4;
    }
    return tets;
}
template <typename T, size_t dim_>
int interp_pts_in_tets(const Eigen::Matrix<T, dim_, -1>& v, const Eigen::Matrix<int, 4, -1>& tet, const Eigen::Matrix<T, dim_, -1>& pts, Eigen::SparseMatrix<T>& coef);

}  // namespace PhysIKA

#endif

// int interp_pts_in_tets(const matd_t &v, const mati_t &tet, const matd_t &pts, csc_t &coef)
// {
//   const size_t tn = tet.size(2), pn = pts.size(2);

//   vector<double*> pv(tn);
//   Eigen::Matrix<T, -1, -1> tet_center(3, tn); {
//     for(int i = 0; i < tn; ++i) {
//       tet_center(colon(), i) = v(colon(), tet(colon(), i))*Eigen::Matrix<T, -1, -1>::Ones(4, 1)/4;
//       pv[i] = &tet_center(0, i);
//     }
//   }

//   std::auto_ptr<ANNkd_tree> kdt(new ANNkd_tree(&pv[0], tn, v.size(1), 32));
//   matrix<Eigen::Matrix<T, -1, -1> > bary_op(tn); {
//     for(int i = 0; i < tn; ++i) {
//       Eigen::Matrix<T, -1, -1> v44 = Eigen::Matrix<T, -1, -1>::Ones(4, 4);
//       for(int j = 0; j < 4; ++j)
//         v44(colon(0, 2), j) = v(colon(), tet(j, i));
//       inv(v44);
//       bary_op[i] = v44;
//     }
//     cout << "create bary-coor operators success." << endl;
//   }

//   vector<Triplet<double>> trips;

//   Eigen::Matrix<T, -1, -1> pt, w;
//   const int ave_k = 40, iter_n = 4;
//   const int max_k = static_cast<int>(40*floor(pow(2.0, iter_n)+0.5));
//   Eigen::Matrix<T, -1, -1> dist(max_k);
//   Eigen::Matrix<int, -1, -1> idx(max_k);
//   double min_good = 1;
//   int outside_cnt = 0;

//   for(int pi = 0; pi < pn; ++pi) {
//     if((pi%1000) == 0)
//       cerr << "process " << pi << endl;

//     pt = pts(colon(), pi);
//     pair<int, double> best_t(-1, -10);

//     for(int ki = 0, k = ave_k; ki < iter_n && k < max_k; ++ki, k*=2) {
//       if(k > max_k)
//         k = max_k;
//       const double r2 = 1e1;
//       kdt->annkSearch(&pt[0], max_k, &idx[0], &dist[0], 1e-10);
//       for(int ti = (k > 40)?k/2:0; ti < k; ++ti) {
//         int t_idx = idx[ti];
//         w = bary_op[t_idx](colon(0, 3), colon(0, 2))*pt + bary_op[t_idx](colon(), 3);
//         double good = std::min(w);
//         if(best_t.second < good) {
//           best_t.second = good;
//           best_t.first = t_idx;
//         }
//         if(best_t.second >= 0)
//           break;
//       }
//       if(best_t.second >= 0)
//         break;
//     }

//     if(best_t.second < 0)
//       ++outside_cnt;
//     if(best_t.second < min_good)
//       min_good = best_t.second;
//     if(best_t.first < 0) {
//       cout << "Wow, very bad point!!" << endl;
//       return __LINE__;
//     }

//     w = bary_op[best_t.first](colon(0, 3), colon(0, 2))*pt + bary_op[best_t.first](colon(), 3);

//     if(fabs(std::sum(w)-1) > 1e-9) {

//       cout << "strange weight." << trans(w);
//       cout << "sum : " << std::sum(w) << endl;
//     }
//     trips.push_back(Triplet<double>(tet(0, best_t.first), pi, w[0]));
//     trips.push_back(Triplet<double>(tet(1, best_t.first), pi, w[1]));
//     trips.push_back(Triplet<double>(tet(2, best_t.first), pi, w[2]));
//     trips.push_back(Triplet<double>(tet(3, best_t.first), pi, w[3]));
//   }
//   cout << "outside pt num is: " << outside_cnt << " min_good is: " << min_good << endl;
//   coef.resize(v.size(2), pn);
//   coef.reserve(trips.size());
//   coef.setFromTriplets(trips.begin(), trips.end());
//   return 0;
// }
