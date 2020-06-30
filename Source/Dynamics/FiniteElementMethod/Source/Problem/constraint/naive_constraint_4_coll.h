/** -*- mode: c++ -*-
 * @file naive_constraint_4_coll.h
 * @author LamKamhang (Cool_Lam@outlook.com)
 * @brief A Documented file.
 * @version 1.0
 * @date Fri Apr 17 22:22:44 CST 2020
 *
 * Detailed description
 *
 *
 * @copyright Copyright (c) 2020
 */
#ifndef NAIVE_CONSTRAINT_4_COLL_H
#define NAIVE_CONSTRAINT_4_COLL_H

#include "constraints.h"
#include <vector>

#include "Common/logger/log_utils.h"

namespace PhysIKA{
  template<typename T>
  class collision_detector;

  template<typename T>
  class plane_detector;

  template<typename T>
  class naive_constraint_4_coll: public constraint_4_coll<T>
  {
    using id_t = unsigned int;
    using sMat_t = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using Tri = Eigen::Triplet<T>;
    using Vec_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using mesh_t = Eigen::Matrix<id_t, Eigen::Dynamic, 1>;
    using nods_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  public:
    naive_constraint_4_coll();
    naive_constraint_4_coll(const std::vector<mesh_t> &meshes,
                            const std::vector<nods_t> &nods,
                            char axis = 'z');

    id_t add_model(const id_t *tris, size_t tris_num,
                   const T* nods, size_t nods_num);
    id_t add_model(const mesh_t &mesh, const nods_t &nods);

    void add_detector(std::shared_ptr<collision_detector<T>> &detector);

    virtual size_t Nx() const { return total_nods_dim; }
    virtual size_t Nf() const { return total_coll_num; }

    // return Jx - c.
    virtual int Val(const T *x, T *val) const;
    virtual int Jac(const T *x, const size_t off,
                    std::vector<Eigen::Triplet<T>> *jac) const;

    virtual int update(const T* x);
    virtual bool verify_no_collision(const T* x_new);
    int update(const std::vector<nods_t> &nods);
    bool verify_no_collision(const std::vector<nods_t> &nods);

  private:
    void init();
    void clear_collision_info();

  private:
    // combine all meshes into a vector.
    mesh_t all_meshes;
    // combine all nods into a vector.
    nods_t all_nods;
    // each model's mesh offset in all_meshes.
    std::vector<size_t> each_mesh_offset;
    // each model's nods offset in all_nods.
    std::vector<size_t> each_nods_offset;
    // collision constraint as triplets.
    std::vector<Tri> triplets;
    // collision right hand side.
    std::vector<T> c;
    // Attention::total_vtx_dim is not only total number of vtxs,
    //            total_vtx_dim is dim * (num of vtxs),
    //            which means 3 times of num of vtxs.
    size_t total_mesh_dim, total_nods_dim, total_coll_num;
    // Determine whether each model has a collision
    // may be used in future optimization work.
    // std::vector<bool> each_model_collision_flag;

    // use detector to do collision detect
    friend class plane_detector<T>;
    std::vector<std::shared_ptr<collision_detector<T>>> detectors;
  };

  template<typename T>
  class collision_detector
  {
  public:
    virtual ~collision_detector() = default;
    virtual bool detect(
      naive_constraint_4_coll<T> &coll,
      const T* x_new) = 0;
  };

  template<typename T>
  class plane_detector: public collision_detector<T>
  {
    using id_t = unsigned int;
    using sMat_t = Eigen::SparseMatrix<T, Eigen::RowMajor>;
    using Tri = Eigen::Triplet<T>;
    using Vec_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
    using mesh_t = Eigen::Matrix<id_t, Eigen::Dynamic, 1>;
    using nods_t = Eigen::Matrix<T, Eigen::Dynamic, 1>;
  public:
    plane_detector(const char axis = 'z')
      : offset(axis - 'x')
      {}
    virtual bool detect(
      naive_constraint_4_coll<T> &coll,
      const T* x_new);
  private:
    const char offset;
  };
} // namespace PhysIKA

////////////////////////////////////////////////////////////////////////
//                    template implementation                         //
////////////////////////////////////////////////////////////////////////
namespace PhysIKA {
  template<typename T>
  void naive_constraint_4_coll<T>::init() {
    all_meshes.resize(0);
    all_nods.resize(0);
    each_mesh_offset.clear();
    each_nods_offset.clear();
    // each_model_collision_flag.clear();
    triplets.clear();
    c.clear();
    total_mesh_dim = 0;
    total_nods_dim = 0;
    total_coll_num = 0;
  }

  template<typename T>
  void naive_constraint_4_coll<T>::clear_collision_info() {
    // each_model_collision_flag.assign(
    //   each_model_collision_flag.size(), false);
    triplets.clear();
    c.clear();
    total_coll_num = 0;
  }

  template<typename T>
  naive_constraint_4_coll<T>::naive_constraint_4_coll()
  {init();}

  template<typename T>
  naive_constraint_4_coll<T>::naive_constraint_4_coll(
    const std::vector<mesh_t> &meshes,
    const std::vector<nods_t> &nods,
    char axis)
  {
    init();
    detectors.emplace_back(
      std::make_shared<plane_detector<T> >(axis));
    assert (meshes.size ()  == nods.size ());
    size_t model_num = meshes.size ();
    for (size_t i = 0; i < model_num; ++i) {
      // this->model_nods_offset.emplace_back (total_nods_dim);
      // total_nods_dim += nods[i].size();
      add_model(meshes[i], nods[i]);
    }
    debug_msg("total_mesh_dim: %lu, total_nods_dim:%lu",
              total_mesh_dim, total_nods_dim);
  }

  template<typename T>
  id_t naive_constraint_4_coll<T>::add_model(
    const id_t *tris, size_t tris_num,
    const T* nods, size_t nods_num)
  {
    Eigen::Map<const mesh_t> local_mesh (tris, tris_num, 1);
    Eigen::Map<const nods_t> local_nods (nods, nods_num, 1);
    return add_model (local_mesh, local_nods);
  }

  template<typename T>
  id_t naive_constraint_4_coll<T>::add_model(
    const mesh_t &mesh, const nods_t &nods)
  {
    id_t id = each_mesh_offset.size();

    mesh_t mesh_joined(all_meshes.size() + mesh.size());
    mesh_joined << all_meshes, mesh;
    all_meshes = mesh_joined;

    nods_t nods_joined(all_nods.size() + nods.size());
    nods_joined << all_nods, nods;
    all_nods = nods_joined;

    each_mesh_offset.emplace_back(total_mesh_dim);
    total_mesh_dim += mesh.size();

    each_nods_offset.emplace_back(total_nods_dim);
    total_nods_dim += nods.size();

    // each_model_collision_flag.push_back(false);
    return id;
  }

  template<typename T>
  int naive_constraint_4_coll<T>::Val(
    const T *x, T *val) const
  {
    if (val == NULL) {
      return 1;
    } else {
      Eigen::Map<Vec_t> V(val, total_coll_num, 1);
      Eigen::Map<const Vec_t> C(c.data(), total_coll_num, 1);
      // make it more easy to return -c.
      if (x == NULL) {
        V = -C;
      } else {
        Eigen::Map<const nods_t> X(x, total_nods_dim, 1);
        sMat_t J(total_coll_num, total_nods_dim);
        J.setFromTriplets(triplets.begin(), triplets.end());
        V = J*X - C;
      }
      return 0;
    }
  }

  template<typename T>
  int naive_constraint_4_coll<T>::Jac(
    const T *, const size_t off,
    std::vector<Eigen::Triplet<T>> *jac) const
  {
    if (jac == NULL) {
      return 1;
    } else {
      for (auto & tpl: triplets) {
        jac->emplace_back(tpl.row()+off, tpl.col(), tpl.value());
      }
      return 0;
    }
  }

  template<typename T>
  int naive_constraint_4_coll<T>::update(
    const std::vector<nods_t> &nods)
  {
    for (size_t i = 0; i < nods.size(); ++i) {
      all_nods.segment(each_nods_offset[i], nods[i].size()) = nods[i];
    }
    return 0;
  }

  template<typename T>
  int naive_constraint_4_coll<T>::update(const T* x)
  {
    if (x == NULL) {
      return 1;
    } else {
      Eigen::Map<const nods_t> X(x, total_nods_dim, 1);
      all_nods = X;
      return 0;
    }
  }

  template<typename T>
  bool naive_constraint_4_coll<T>::verify_no_collision(
    const std::vector<nods_t> &nods)
  {
    assert(nods.size() == each_nods_offset.size());
    nods_t nods_join(total_nods_dim);
    for (size_t i = 0; i < nods.size(); ++i) {
      nods_join.segment(each_nods_offset[i], nods[i].size()) = nods[i];
    }
    return verify_no_collision(nods_join.data());
  }

  template<typename T>
  bool naive_constraint_4_coll<T>::verify_no_collision(
    const T* x_new)
  {
    clear_collision_info();
    for (auto & detector : detectors) {
      detector->detect(*this, x_new);
    }
    return total_coll_num == 0;
  }

  template<typename T>
  void naive_constraint_4_coll<T>::add_detector(
    std::shared_ptr<collision_detector<T>> &detector) {
    detectors.emplace_back(detector);
  }

  template<typename T>
  bool plane_detector<T>::detect(
    naive_constraint_4_coll<T> &coll,
    const T* x_new)
  {
    bool res = false;
    Eigen::Map<const nods_t> X(x_new, coll.total_nods_dim, 1);
    for (size_t i = 0; i < coll.total_nods_dim / 3; ++i) {
      int v_idx = 3*i+offset;
      if (X[v_idx] < 0) {
        res = true;
        coll.triplets.emplace_back(
          coll.total_coll_num, v_idx, 1);
        coll.c.emplace_back(0);
        coll.total_coll_num++;
      }
    }
    return res;
  }
} // namespace PhysIKA

#endif /* NAIVE_CONSTRAINT_4_COLL_H */
