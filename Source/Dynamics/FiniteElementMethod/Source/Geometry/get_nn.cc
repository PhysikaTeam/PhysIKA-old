#include "get_nn.h"
#include "geometry.h"

#include <cmath>
#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <Eigen/Geometry>
using namespace std;
using namespace Eigen;

namespace marvel {

spatial_hash::spatial_hash(const MatrixXd& points_, const size_t& nn_num_)
    : points(points_), points_num(points_.cols()), nn_num(nn_num_)
{
    //set hash parameter
    size_t table_size = size_t(floor(pow(points.cols(), 0.5)));
    //build hash_map
    points_hash = unordered_multimap<Vector3i, size_t>(table_size);
    hash_NNN();
}

int spatial_hash::get_shell(const Eigen::Vector3i& query, const int& radi, std::vector<Vector3i>& shell) const
{
    assert(radi > -1);
    //init
    shell.clear();
    int res = 0;

    auto loop = [&](const int& face, const int& face_axis, const int& radi_1, const int& radi_2) {
        if (face <= max_id(face_axis) && face >= min_id(face_axis))
        {
            int axis_1 = (face_axis + 1) % 3, axis_2 = (face_axis + 2) % 3;
            for (int j = query(axis_1) - radi_1; j < query(axis_1) + radi_1 + 1; ++j)
            {
                for (int k = query(axis_2) - radi_2; k < query(axis_2) + radi_2 + 1; ++k)
                {
                    Vector3i one_grid;
                    one_grid[face_axis] = face;
                    one_grid[axis_1]    = j;
                    one_grid[axis_2]    = k;
                    shell.push_back(one_grid);
                }
            }
            return 0;
        }
        else
        {
            return 1;
        }
    };
    if (radi > 0)
    {
        int touch_time = 0;
        for (size_t i = 0; i < 6; ++i)
        {

            touch_time += loop(i % 2 == 0 ? query(i / 2) - radi : query(i / 2) + radi, i / 2, i < 4 ? radi : radi - 1, i < 2 ? radi : radi - 1);
        }
        if (touch_time == 6)
            res = -1;
    }
    else
    {
        shell.push_back(query);
    }
    return res;
}

int spatial_hash::find_NN(const size_t& point_id, vector<pair_dis>& NN_cand)
{
    return std::move(find_NN(point_id, NN_cand, nn_num));
}

int spatial_hash::find_NN(const size_t& point_id, vector<pair_dis>& NN_cand, const size_t& nn_num_)
{

    size_t cand_num  = 0;
    int    grid_delt = 0;
    // bool once_more = false;
    int once_more = 0;
    int touch_bd  = 0;
    //count for grid_delt;
    do
    {
        vector<Vector3i> shell;
        touch_bd = get_shell(points_dis.col(point_id), grid_delt, shell);
        for (auto& grid : shell)
        {
            auto range = points_hash.equal_range(grid);
            if (range.first != range.second)
            {
                for_each(range.first, range.second, [&](decltype(points_hash)::value_type& one_point) {
                    double dis = (points.col(point_id) - points.col(one_point.second)).norm();
                    NN_cand.push_back({ one_point.second, dis });
                });
            }
        }
        ++grid_delt;
        if (NN_cand.size() > nn_num_ + 2)
            // once_more = !once_more;
            once_more++;
        if (touch_bd)
            break;

    } while (NN_cand.size() < nn_num_ + 2 || once_more < 2);
    return 0;
}

int spatial_hash::hash_NNN()
{
    //init
    NN.setZero(nn_num, points_num);
    points_hash.clear();
    //set parameter

    double   cell_num = pow(points.cols() / float(nn_num), 1.0 / 3);
    MatrixXd bdbox(3, 2);
    build_bdbox(points, bdbox);
    cell_size = (bdbox.col(1) - bdbox.col(0)) / cell_num;
    //generate discretized 3D position
    points_dis = MatrixXi(3, points_num);
    for (size_t i = 0; i < points.rows(); ++i)
    {
        points_dis.row(i) = floor(points.row(i).array() / cell_size(i)).cast<int>();
    }

    max_id = { points_dis.row(0).maxCoeff(), points_dis.row(1).maxCoeff(), points_dis.row(2).maxCoeff() };
    min_id = { points_dis.row(0).minCoeff(), points_dis.row(1).minCoeff(), points_dis.row(2).minCoeff() };

    //insert elements

    for (size_t i = 0; i < points_num; ++i)
    {
        points_hash.insert({ points_dis.col(i), i });
    }

    return 0;
}

const MatrixXi& spatial_hash::get_NN(const size_t& nn_num_)
{
    auto useless_sup_radi = get_sup_radi(nn_num_);

    return NN;
}
const MatrixXi& spatial_hash::get_NN()
{
    return std::move(get_NN(nn_num));
}

const VectorXi spatial_hash::get_NN(const Vector3d& query, const size_t& nn_num_)
{
    //init data
    assert(nn_num_ < points_num);
    VectorXi         near_nei = VectorXi::Zero(nn_num_);
    vector<pair_dis> NN_cand;

    size_t         cand_num    = 0;
    int            grid_delt   = 0;
    int            once_more   = 0;
    const Vector3i center_grid = floor(query.array() / cell_size.array()).cast<int>();
    do
    {
        vector<Vector3i> shell;
        get_shell(center_grid, grid_delt, shell);
        for (auto& grid : shell)
        {
            auto range = points_hash.equal_range(grid);
            if (range.first != range.second)
            {
                for_each(range.first, range.second, [&](decltype(points_hash)::value_type& one_point) {
                    double dis = (query - points.col(one_point.second)).norm();
                    NN_cand.push_back({ one_point.second, dis });
                });
            }
        }
        ++grid_delt;
        if (NN_cand.size() > nn_num_ + 2)
            once_more++;
    } while (NN_cand.size() < nn_num_ + 2 || once_more < 2);

    sort(NN_cand.begin(), NN_cand.end(), [](const pair_dis& a, const pair_dis& b) { return a.dis < b.dis; });

    for (size_t j = 0; j < nn_num_; ++j)
    {
        near_nei(j) = NN_cand[j].n;
    }
    return near_nei;
}

const Eigen::MatrixXi spatial_hash::get_four_noncoplanar_NN(const Eigen::MatrixXd& nods)
{
    MatrixXi near_nei = MatrixXi::Zero(4, nods.cols());
    // #pragma omp parallel for
    for (size_t i = 0; i < nods.cols(); ++i)
    {

        auto     ver_nn_num = 4;
        Vector2i two_index;
        two_index << 2, 3;
        bool is_co_line = true, is_co_plane = true;
        do
        {
            auto     NN_index = get_NN(nods.col(i), ver_nn_num);
            Vector3d V1       = (points.col(NN_index(1)) - points.col(NN_index(0))).normalized();
            Vector3d V2       = (points.col(NN_index(two_index(0))) - points.col(NN_index(0))).normalized();
            Vector3d cross_   = V1.cross(V2);
            if (cross_.squaredNorm() < 1e-5)
            {
                two_index(0)++;
                two_index(1)++;
                ver_nn_num++;
                is_co_line = true;
            }
            else
            {
                is_co_line = false;

                do
                {
                    Vector3d V3 = (points.col(NN_index(two_index(1))) - points.col(NN_index(0))).normalized();
                    if (fabs(cross_.dot(V3)) < 1e-5)
                    {
                        two_index(1)++;
                        ver_nn_num++;
                        NN_index = get_NN(nods.col(i), ver_nn_num);
                        // assert(ver_nn_num < 10);
                        is_co_plane = true;
                    }
                    else
                    {
                        is_co_plane    = false;
                        near_nei(0, i) = NN_index(0);
                        near_nei(1, i) = NN_index(1);
                        near_nei(2, i) = NN_index(two_index(0));
                        near_nei(3, i) = NN_index(two_index(1));
                        break;
                    }
                } while (is_co_plane);
            }
        } while (is_co_line);
    }
    return near_nei;
}

const VectorXd& spatial_hash::get_sup_radi()
{
    return std::move(get_sup_radi(nn_num));
}

const VectorXd& spatial_hash::get_sup_radi(const size_t& nn_num_)
{
    assert(points.cols() >= nn_num_);

    //init data

    sup_radi.setZero(points_num);

#pragma omp parallel for
    for (size_t i = 0; i < points_num; ++i)
    {
        vector<pair_dis> NN_cand;
        find_NN(i, NN_cand, nn_num_ + 2);
        sort(NN_cand.begin(), NN_cand.end(), [](const pair_dis& a, const pair_dis& b) { return a.dis < b.dis; });
        for (size_t j = 0; j < nn_num_; ++j)
        {
            NN(j, i) = NN_cand[j].n;
            sup_radi[i] += NN_cand[j].dis;
        }
    }
    sup_radi *= 3.0 / nn_num_;
    return sup_radi;
}
int spatial_hash::get_friends(const Vector3d& query, const double& sup_radi, vector<size_t>& friends, bool is_sample) const
{

    friends.clear();
    int            grid_delt   = static_cast<int>(ceil(sup_radi / cell_size.col(0).minCoeff())) + 1;
    const Vector3i center_grid = floor(query.array() / cell_size.array()).cast<int>();
    for (size_t i = 0; i < grid_delt; ++i)
    {
        vector<Vector3i> shell;
        get_shell(center_grid, i, shell);
        for (auto& one_grid : shell)
        {
            auto range = points_hash.equal_range(one_grid);
            if (range.first != range.second)
            {
                for_each(range.first, range.second, [&](const decltype(points_hash)::value_type& one_point) {
                    double dis = (points.col(one_point.second) - query).norm();
                    if (dis < sup_radi)
                        friends.push_back(one_point.second);
                });
            }
        }
    }
    assert(friends.size() > nn_num);
    return 0;
}
// int spatial_hash::update_points(const Eigen::MatrixXd &points_){
//   points = points_;
//   hash_NNN();
//   return 0;
// }

}  // namespace marvel
