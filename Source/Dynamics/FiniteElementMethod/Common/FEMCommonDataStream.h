/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: data stream core helper.
 * @version    : 1.0
 */
#pragma once

#include <unordered_map>
#include <Eigen/Dense>
#include <Eigen/Sparse>
namespace PhysIKA {
/**
 * data stream core class, store some data related to the optimization problem
 *
 */
template <typename T, size_t dim_>
class dat_str_core
{
public:
    using SMP_TYPE = Eigen::SparseMatrix<T, Eigen::RowMajor>;

    virtual ~dat_str_core() {}
    dat_str_core(const size_t& dof, const bool hes_is_const = true);

    /**
     * @brief Set the zero object
     * 
     * @return int 
     */
    int set_zero();

    /**
     * @brief reserve the hes storage.
     * !!!!!!!WARNING!!!!!!!!!:   reserve enough space
     * 
     * @param nnzs 
     * @return int 
     */
    int hes_reserve(const Eigen::VectorXi& nnzs);

    /**
     * @brief compress hes triplets.
     * 
     * @return int 
     */
    int hes_compress();

    /**
     * @brief hessian add diagonal.
     * 
     * @param time 
     * @return int 
     */
    int hes_add_diag(const size_t& time);

    /**
     * @brief Set the From Triplets object
     * 
     * @return int 
     */
    int setFromTriplets();

    /**
     * @brief save val to data stream.
     * 
     * @param val 
     * @return int 
     */
    int save_val(const T& val);

    /**
     * @brief save gradient to data stream.
     * 
     * @param gra 
     * @return int 
     */
    int save_gra(const Eigen::Matrix<T, Eigen::Dynamic, 1>& gra);

    /**
     * @brief save gradient to data stream.
     * 
     * @param pos position
     * @param point_gra position gradient.
     * @return int 
     */
    int save_gra(const size_t& pos, const Eigen::Matrix<T, dim_, 1>& point_gra);

    /**
     * @brief save gradient to data stream.
     * 
     * @param pos 
     * @param one_gra 
     * @return int 
     */
    int save_gra(const size_t& pos, const T& one_gra);

    /**
     * @brief save gradient to data stream.
     * 
     * @tparam Derived 
     * @param pos 
     * @param point_gra 
     * @return int 
     */
    template <typename Derived>
    int save_gra(const size_t& pos, const Eigen::MatrixBase<Derived>& point_gra)
    {
        for (size_t d = 0; d < dim_; ++d)
        {
#pragma omp atomic
            gra_(dim_ * pos + d) += point_gra(d);
        }
        return 0;
    }

    /**
     * @brief save hessian to data stream.
     * 
     * @param m 
     * @param n 
     * @param loc_hes 
     * @return int 
     */
    int save_hes(const size_t& m, const size_t& n, const Eigen::Matrix<T, dim_, dim_>& loc_hes);

    /**
     * @brief save hessian to data stream.
     * 
     * @param row 
     * @param col 
     * @param value 
     * @return int 
     */
    int save_hes(const size_t& row, const size_t& col, const T& value);

    /**
     * @brief Set the hes zero after pre compute object
     * 
     * @return int 
     */
    int set_hes_zero_after_pre_compute();

    /**
     * @brief Get the val object
     * 
     * @return const T 
     */
    const T get_val() const;

    /**
     * @brief Get the gra object
     * 
     * @return const Eigen::Matrix<T, Eigen::Dynamic, 1>& 
     */
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& get_gra() const;

    /**
     * @brief Get the hes object
     * 
     * @return const SMP_TYPE& 
     */
    const SMP_TYPE& get_hes() const;

    /**
     * @brief Get the dof object
     * 
     * @return const size_t 
     */
    const size_t get_dof() const;
    //TODO:add Perfect Forwardincg

private:
    const size_t                        dof_;
    const size_t                        whole_dim_;
    T                                   val_;
    Eigen::Matrix<T, Eigen::Dynamic, 1> gra_;
    SMP_TYPE                            hes_;
    Eigen::Matrix<T, Eigen::Dynamic, 1> all_one_;
    bool                                has_pre_compute_hes_{ false };
    std::vector<Eigen::Triplet<T>>      trips;
    bool                                hes_is_const_;
    std::unordered_map<size_t, T*>      hes_ref_;
};

}  // namespace PhysIKA
