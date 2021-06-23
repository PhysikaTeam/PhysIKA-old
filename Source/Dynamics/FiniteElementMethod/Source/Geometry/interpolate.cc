#include "interpolate.h"
#include <list>
namespace PhysIKA{
using namespace Eigen;
using namespace std;
template<typename FLOAT, size_t dim_>
Vector4i get_noncoplanar_tet(const Eigen::Matrix<FLOAT, dim_, -1> &v,
                             const std::vector<size_t> &neigh_vert_idx){
  using Vector3F = Matrix<FLOAT, 3, 1>;
  if(neigh_vert_idx.size() < 4){
    cerr << "neigh_vert_id size < 4! " << endl;
    exit(EXIT_FAILURE);
  }

  list<size_t> vert_left(neigh_vert_idx.begin(), neigh_vert_idx.end());

  Vector4i tet;
  tet(0) = neigh_vert_idx[0];
  vert_left.pop_front();
  tet(1) = neigh_vert_idx[1];
  vert_left.pop_front();
  //find third point
  const Vector3F edge0 = v.col(tet(1)) - v.col(tet(0));

  bool co_linear = true;{
    for(auto it = vert_left.begin(); it!= vert_left.end(); ++it){
      const Vector3F edge1 = v.col(*it) - v.col(tet(0));
	  const auto sin_a = fabs(edge1.cross(edge0).norm()) / (edge1.norm() * edge0.norm());
      if(sin_a > 1e-3){
        tet(2) =  *it;
        co_linear = false;
        vert_left.erase(it);
        break;
      }
    }
    if(co_linear){
      cerr << "All the points are in one line!." << endl;
      exit(EXIT_FAILURE);
    } 
  }

  const Vector3F edge1 = v.col(tet(2)) - v.col(tet(0));
  const auto area = edge0.cross(edge1).norm();

  //fint 4th point
  bool co_planar = true;{
    Matrix<FLOAT, 4, 4> tet_geo= Matrix<FLOAT, 4, 4>::Ones();
    tet_geo.block<3,3>(0, 0) = v(all, tet.topRows<3>());

    for(auto it = vert_left.begin(); it!=vert_left.end(); ++it){
     tet_geo.block<3,1>(0,3) = v.col(*it);
	 const auto height = fabs(tet_geo.determinant()) / area;
	 auto longest_edge_length = 0.0; {
		 Vector3F lengths = Vector3F::Zero();
		 for (int i = 0; i < 3; ++i)
			 lengths(i) = (v.col(*it) - v.col(tet(i))).norm();
		 longest_edge_length = lengths.maxCoeff();
	 }
	 const auto sin_beta = height / longest_edge_length;
     if(sin_beta > 1e-3){
       co_planar = false;
       tet(3) = *it;
       vert_left.erase(it);
       break;
     }
    }
   if(co_planar){
     cerr << "All the points are in one plane" << endl;
     exit(EXIT_FAILURE);
   }
  }
  return tet;
  
}

template Vector4i get_noncoplanar_tet(const Eigen::Matrix<float, 3, -1> &v, const std::vector<size_t> &neigh_vert_idx);
template Vector4i get_noncoplanar_tet(const Eigen::Matrix<double, 3, -1>&v, const std::vector<size_t> &neigh_vert_idx);

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


	
    // std::unique_ptr<my_kd_tree_t> kdt(new my_kd_tree_t(dim_, pv, nanoflann::KDTreeSingleIndexAdaptorParams(50 /* max leaf */)));
    // kdt->buildIndex();

    MatrixXd shv = v.cast<double>();
    marvel::spatial_hash sh(shv);

    const int pt_num = pts.cols();
    Eigen::Matrix<T, -1, 1> pt, weights;
    const int neigh_vert_num = 50;
    std::vector<T> neigh_vert_dis(neigh_vert_num);
    std::vector<size_t> neigh_vert_idx(neigh_vert_num);
    std::vector<Eigen::Triplet<T>> trips;
    for (int pi = 0; pi < pt_num; ++pi)
    {
      pt = pts.col(pi);
      // kdt->knnSearch(
      //   &pt[0], neigh_vert_num, neigh_vert_idx.data(), neigh_vert_dis.data());


	  const Vector3d pt_double = pt.cast<double>();
    const auto nn_vert_idx = sh.get_NN(pt_double, neigh_vert_num);
	neigh_vert_idx.clear();
    neigh_vert_idx.resize(nn_vert_idx.size());
    for(size_t i = 0; i < nn_vert_idx.size(); ++i){
      neigh_vert_idx[i] = nn_vert_idx(i);
	  neigh_vert_dis[i] = (pt - v.col(neigh_vert_idx[i])).norm();
    }
	

	auto one_tet = get_noncoplanar_tet<T, dim_>(v, neigh_vert_idx);


      int near_v_idx = 0;
      int further_v_idx = 4;
      Eigen::Matrix<T, -1, -1> v44 = Eigen::Matrix<T, -1, -1>::Ones(4, 4);
        for(int j = 0; j < 4; ++j)
          v44.block(0, j, 3, 1) = v.col(one_tet(j));

      // while (true)
      // {
      //   for(int j = 0; j < 4; ++j)
      //     v44.block(0, j, 3, 1) = v.col(neigh_vert_idx[j]);
      //   if (fabs(v44.determinant()) > 1e-7)
      //     break;
      //   std::swap(neigh_vert_idx[further_v_idx], neigh_vert_idx[near_v_idx]);
      //   ++further_v_idx;
      //   if (further_v_idx == neigh_vert_num)
      //   {
      //     further_v_idx = 4;
      //     std::swap(neigh_vert_idx[near_v_idx], neigh_vert_idx[further_v_idx]);
      //     ++near_v_idx;
      //   }
      //   if (near_v_idx == 4)
      //   {
      //     std::cout << "error int point interp" << std::endl;
		  // {
      //   for(size_t i = 0; i < neigh_vert_idx.size(); ++i){
      //     cout << neigh_vert_idx[i] << " ";
      //   }
      //   cout << endl;
      //   cout << "pi " << pi << "\npt:\n" << pt << endl;

		  // }
      //     return __LINE__;
      //   }
      // }
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

template int interp_pts_in_point_cloud(const Eigen::Matrix<float, 3,-1> &v, const Eigen::Matrix<float, 3, -1> &pts, Eigen::SparseMatrix<float> &coef);
template int interp_pts_in_point_cloud(const Eigen::Matrix<double, 3,- 1> &v,const Eigen::Matrix<double, 3, -1> &pts, Eigen::SparseMatrix<double> &coef);

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
            // if((pi%1000) == 0)
            // std::cerr << "process " << pi << std::endl;

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
    template int interp_pts_in_tets(const Eigen::Matrix<float, 3, -1> &v,
                                    const Eigen::Matrix<int, 4, -1> &tet,
                                    const Eigen::Matrix<float, 3, -1> &pts,
                                    Eigen::SparseMatrix<float> &coef);
    template int interp_pts_in_tets(const Eigen::Matrix<double, 3, -1> &v,
                                    const Eigen::Matrix<int, 4, -1> &tet,
                                    const Eigen::Matrix<double, 3, -1> &pts,
                                    Eigen::SparseMatrix<double> &coef);
;

  }
