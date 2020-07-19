#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "nanoflann/nanoflann.hpp"
#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <Eigen/Sparse>
// #include <ANN/ANN.h>
#include "Common/eigen_ext.h"
#include <iostream>
#include <vector>
#include <memory>

namespace PhysIKA {
    template <typename T>
    struct PointCloud
    {
        struct Point
        {
            T  x,y,z;
        };

        std::vector<Point>  pts;

        // Must return the number of data points
        inline size_t kdtree_get_point_count() const { return pts.size(); }

        // Returns the dim'th component of the idx'th point in the class:
        // Since this is inlined and the "dim" argument is typically an immediate value, the
        //  "if/else's" are actually solved at compile time.
        inline T kdtree_get_pt(const size_t idx, const size_t dim) const
        {
            if (dim == 0) return pts[idx].x;
            else if (dim == 1) return pts[idx].y;
            else return pts[idx].z;
        }

        // Optional bounding-box computation: return false to default to a standard bbox computation loop.
        //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
        //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const { return false; }

    };


  template<typename T, size_t dim_>
  int interp_pts_in_point_cloud(const Eigen::Matrix<T, dim_, -1> &v, const Eigen::Matrix<T, dim_, -1> &pts, Eigen::SparseMatrix<T> &coef)
  {
    const int vert_num = v.cols();
    PointCloud<T> pv;
    pv.pts.resize(vert_num);
    for (int i = 0; i < vert_num; ++i)
    {
      pv.pts[i].x = v(0, i);
      pv.pts[i].y = v(1, i);
      if (dim_ == 3)
        pv.pts[i].z = v(2, i);
    }

    // construct a kd-tree index:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<T, PointCloud<T> > ,
      PointCloud<T>,
      dim_ /* dim */
      > my_kd_tree_t;
    
    std::unique_ptr<my_kd_tree_t> kdt(new my_kd_tree_t(dim_, pv, nanoflann::KDTreeSingleIndexAdaptorParams(32 /* max leaf */)));
    kdt->buildIndex();

    const int pt_num = pts.cols();
    Eigen::Matrix<T, -1, 1> pt, weights;      
    const int neigh_vert_num = 30;
    std::vector<T> neigh_vert_dis(neigh_vert_num);
    std::vector<size_t> neigh_vert_idx(neigh_vert_num);
    std::vector<Eigen::Triplet<T>> trips;
    for (int pi = 0; pi < pt_num; ++pi)
    {
      pt = pts.col(pi);
      kdt->knnSearch(
        &pt[0], neigh_vert_num, neigh_vert_idx.data(), neigh_vert_dis.data());

      int near_v_idx = 0;
      int further_v_idx = 4;
      Eigen::Matrix<T, -1, -1> v44 = Eigen::Matrix<T, -1, -1>::Ones(4, 4);
      while (true)
      {
        for(int j = 0; j < 4; ++j)
          v44.block(0, j, 3, 1) = v.col(neigh_vert_idx[j]);
        if (fabs(v44.determinant()) > 1e-7)
          break;
        std::swap(neigh_vert_idx[further_v_idx], neigh_vert_idx[near_v_idx]);
        ++further_v_idx;
        if (further_v_idx == neigh_vert_num)
        {
          further_v_idx = 4;
          std::swap(neigh_vert_idx[near_v_idx], neigh_vert_idx[further_v_idx]);
          ++near_v_idx;
        }
        if (near_v_idx == 4)
        {
          std::cout << "error int point interp" << std::endl;
          return __LINE__;
        }
      }
      v44 = v44.inverse();
      weights = v44.block(0,0,4,3) * pt + v44.col(3);

      for (int t = 0; t < 4; ++t)
        trips.push_back(Eigen::Triplet<T>(neigh_vert_idx[t], pi, weights[t]));
    }

    coef.resize(v.cols(), pts.cols());
    coef.reserve(trips.size());
    coef.setFromTriplets(trips.begin(), trips.end());
    return 0;
  }
  
    template<typename T, size_t dim_>
    int interp_pts_in_tets(const Eigen::Matrix<T, dim_, -1> &v, const Eigen::Matrix<int, 4, -1> &tet, const Eigen::Matrix<T, dim_, -1> &pts, Eigen::SparseMatrix<T> &coef)
    {
      if (tet.size() == 0)
        return interp_pts_in_point_cloud<T, dim_>(v, pts, coef);
      
      const size_t tn = tet.cols(), pn = pts.cols();

      Eigen::Matrix<int, dim_, 1> all_rows_ = Eigen::Matrix<int, dim_, 1>::LinSpaced(dim_, 0, dim_ -1);

        PointCloud<T> pv;
        pv.pts.resize(tn);
        // std::vector<T*> pv(tn);
        Eigen::Matrix<T, -1, -1> tet_center(3, tn); {
            for(int i = 0; i < tn; ++i) {
                // tet_center(colon(), i) = v(colon(), tet(colon(), i))*ones<double>(4, 1)/4;
                // tet_center.col(i) = indexing(v, all_rows_, tet.col(i)) * Eigen::Matrix<T, -1, -1>::Ones(4, 1) / 4;
                Eigen::Matrix<T, 3, 4> one_tet = indexing(v, all_rows_, tet.col(i));
                tet_center.col(i) = one_tet * Eigen::Matrix<T, 4, 1>::Ones() / 4;
                pv.pts[i].x = tet_center(0, i);
                pv.pts[i].y = tet_center(1, i);
                pv.pts[i].z = tet_center(2, i);
            }
        }

        // construct a kd-tree index:
        typedef nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<T, PointCloud<T> > ,
            PointCloud<T>,
            dim_ /* dim */
        > my_kd_tree_t;


        std::unique_ptr<my_kd_tree_t> kdt(new my_kd_tree_t(dim_, pv, nanoflann::KDTreeSingleIndexAdaptorParams(32 /* max leaf */)));
        kdt->buildIndex();
        // std::unique_ptr<ANNkd_tree> kdt(new ANNkd_tree((ANNpointArray)&pv[0], (int)tn, (int)v.rows(), 32));
        std::vector<Eigen::Matrix<T, 4, 4>, Eigen::aligned_allocator<Eigen::Matrix<T, 4, 4>>> bary_op(tn); {
            for(int i = 0; i < tn; ++i) {
                Eigen::Matrix<T, -1, -1> v44 = Eigen::Matrix<T, -1, -1>::Ones(4, 4);
                for(int j = 0; j < 4; ++j)
                    //v44(colon(0, 2), j) = v(colon(), tet(j, i));
                    v44.block(0, j, 3, 1) = v.col(tet(j, i));
                v44 = v44.inverse();
                bary_op[i] = v44;
            }
            std::cout << "create bary-coor operators success." << std::endl;
        }

        std::vector<Eigen::Triplet<double>> trips;

        Eigen::Matrix<T, -1, 1> pt, w;
        const int ave_k = 40, iter_n = 4;
        const int max_k = static_cast<int>(40*floor(pow(2.0, iter_n)+0.5));
        std::vector<T> dist(max_k);
        std::vector<size_t> idx(max_k);
        double min_good = 1;
        int outside_cnt = 0;

        for(int pi = 0; pi < pn; ++pi) {
            if((pi%1000) == 0)
            std::cerr << "process " << pi << std::endl;

            pt = pts.col(pi);
            std::pair<int, double> best_t(-1, -10);

            for(int ki = 0, k = ave_k; ki < iter_n && k < max_k; ++ki, k*=2) {
            if(k > max_k)
                k = max_k;
            const double r2 = 1e1;
            kdt->knnSearch(&pt[0], max_k, &idx[0], &dist[0]);
            // kdt->annkSearch((ANNpoint)&pt(0), max_k, (ANNidxArray)&idx[0], (ANNdistArray)&dist[0], 1e-10);
            for(int ti = (k > 40)?k/2:0; ti < k; ++ti) {
                int t_idx = idx[ti];
                w = bary_op[t_idx].block(0,0,4,3) *pt + bary_op[t_idx].col(3);
                double good = w.minCoeff();
                if(best_t.second < good) {
                best_t.second = good;
                best_t.first = t_idx;
                }
                if(best_t.second >= 0)
                break;
            }
            if(best_t.second >= 0)
                break;
            }

            if(best_t.second < 0)
            ++outside_cnt;
            if(best_t.second < min_good)
            min_good = best_t.second;
            if(best_t.first < 0) {
                std::cout << "Wow, very bad point!!" << std::endl;
                return __LINE__;
            }

            w = bary_op[best_t.first].block(0, 0, 4, 3) * pt + bary_op[best_t.first].col(3);

            if(std::fabs(w.sum()-1) > 1e-9) {
                // std::cout << "strange weight." << w.transpose();
                // std::cout << "sum : " << w.sum() << std::endl;
            }
            trips.push_back(Eigen::Triplet<double>(tet(0, best_t.first), pi, w[0]));
            trips.push_back(Eigen::Triplet<double>(tet(1, best_t.first), pi, w[1]));
            trips.push_back(Eigen::Triplet<double>(tet(2, best_t.first), pi, w[2]));
            trips.push_back(Eigen::Triplet<double>(tet(3, best_t.first), pi, w[3]));
        }
        std::cout << "outside pt num is: " << outside_cnt << " min_good is: " << min_good << std::endl;
        coef.resize(v.cols(), pn);
        coef.reserve(trips.size());
        coef.setFromTriplets(trips.begin(), trips.end());
        return 0;
    }
}

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
