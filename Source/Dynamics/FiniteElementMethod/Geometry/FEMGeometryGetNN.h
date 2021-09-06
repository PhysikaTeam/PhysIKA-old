#pragma once

#include <vector>
#include <memory>
#include <list>
#include <Eigen/Core>

#include <unordered_map>

namespace std {
template <>
struct hash<Eigen::Vector3i>
{
    typedef size_t          result_type;
    typedef Eigen::Vector3i argument_type;
    size_t                  operator()(const Eigen::Vector3i& key) const
    {
        // return  ( (key(0)*73856093) ^ (key(1)*19349663) ^ (key(2)*83492791) ) % 5999;
        return ((key(0) * 73856093) ^ (key(1) * 19349663) ^ (key(2) * 83492791));
    }
};

}  // namespace std

namespace PhysikaFEM {
struct pair_dis
{
    size_t n;
    double dis;
};

/**
 * @brief FEM Geometry spatial_hash
 * 
 */
class spatial_hash
{
public:
    /**
     * @brief Construct a new spatial hash object
     * 
     * @param points_ 
     * @param nn_num_ 
     */
    spatial_hash(const Eigen::MatrixXd& points_, const size_t& nn_num_ = 10);

    /**
     * @brief Get the NN data
     * 
     * @param nn_num_ 
     * @return const Eigen::MatrixXi& 
     */
    const Eigen::MatrixXi& get_NN(const size_t& nn_num_);

    /**
     * @brief Get the NN data
     * 
     * @return const Eigen::MatrixXi& 
     */
    const Eigen::MatrixXi& get_NN();

    /**
     * @brief find nn_num nearest neighbours of any point
     * 
     * @param query 
     * @param nn_num 
     * @return const Eigen::VectorXi 
     */
    const Eigen::VectorXi  get_NN(const Eigen::Vector3d& query, const size_t& nn_num);

    /**
     * @brief Get the four noncoplanar NN object
     * 
     * @param nods 
     * @return const Eigen::MatrixXi 
     */
    const Eigen::MatrixXi  get_four_noncoplanar_NN(const Eigen::MatrixXd& nods);

    /**
     * @brief Get the sup radi object
     * 
     * @param nn_num_ 
     * @return const Eigen::VectorXd& 
     */
    const Eigen::VectorXd& get_sup_radi(const size_t& nn_num_);

    /**
     * @brief Get the sup radi object
     * 
     * @return const Eigen::VectorXd& 
     */
    const Eigen::VectorXd& get_sup_radi();

    /**
     * @brief Get the shell object
     * 
     * @param query 
     * @param radi 
     * @param shell 
     * @return int 
     */
    int get_shell(const Eigen::Vector3i& query, const int& radi, std::vector<Eigen::Vector3i>& shell) const;
    // int get_friends(const size_t &point_id, const double &sup_radi, std::vector<size_t> &friends) const;
    int get_friends(const Eigen::Vector3d& query, const double& sup_radi, std::vector<size_t>& friends, bool is_sample = true) const;
    // int update_points(const Eigen::MatrixXd &points_);
private:
    const Eigen::MatrixXd                            points;
    size_t                                           nn_num;
    size_t                                           points_num;
    std::unordered_multimap<Eigen::Vector3i, size_t> points_hash;
    Eigen::MatrixXi                                  points_dis;

    Eigen::MatrixXi NN;
    Eigen::VectorXd sup_radi;

    Eigen::Vector3i max_id;
    Eigen::Vector3i min_id;
    /**
     * @brief Find the NN object
     * 
     * @param point_id 
     * @param NN_cand 
     * @return int 
     */
    int             find_NN(const size_t& point_id, std::vector<pair_dis>& NN_cand);
    /**
     * @brief Find the NN object
     * 
     * @param point_id 
     * @param NN_cand 
     * @param nn_num_ 
     * @return int 
     */
    int             find_NN(const size_t& point_id, std::vector<pair_dis>& NN_cand, const size_t& nn_num_);

    /**
     * @brief Initialize hash nnn data 
     * 
     * @return int
     */
    int             hash_NNN();

    Eigen::Vector3d cell_size;
};

}  // namespace PhysikaFEM
