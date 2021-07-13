#ifndef GET_NN_H
#define GET_NN_H

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

namespace marvel {
struct pair_dis
{
    size_t n;
    double dis;
};

class spatial_hash
{
public:
    spatial_hash(const Eigen::MatrixXd& points_, const size_t& nn_num_ = 10);

    const Eigen::MatrixXi& get_NN(const size_t& nn_num_);
    const Eigen::MatrixXi& get_NN();
    //find nn_num nearest neighbours of any point
    const Eigen::VectorXi  get_NN(const Eigen::Vector3d& query, const size_t& nn_num);
    const Eigen::MatrixXi  get_four_noncoplanar_NN(const Eigen::MatrixXd& nods);
    const Eigen::VectorXd& get_sup_radi(const size_t& nn_num_);
    const Eigen::VectorXd& get_sup_radi();

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
    int             find_NN(const size_t& point_id, std::vector<pair_dis>& NN_cand);
    int             find_NN(const size_t& point_id, std::vector<pair_dis>& NN_cand, const size_t& nn_num_);
    int             hash_NNN();

    Eigen::Vector3d cell_size;
};

}  // namespace marvel
#endif
