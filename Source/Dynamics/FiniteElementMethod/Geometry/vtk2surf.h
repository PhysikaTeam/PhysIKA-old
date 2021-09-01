/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: vtk to surface.
 * @version    : 1.0
 */
#ifndef VTK2SURF_H
#define VTK2SURF_H
#include <Eigen/Core>
#include <map>
using namespace Eigen;
using namespace std;

namespace PhysIKA {

// just for triangle face now
// TODO: let it

Vector3i sort_v(const Vector3i& V)
{
    auto sorted = V;
    auto swap   = [&](const size_t& i, const size_t& j) {
        size_t temp = sorted(i);
        sorted(i)   = sorted(j);
        sorted(j)   = temp;
    };
    if (sorted(0) > sorted(1))
        swap(0, 1);
    if (sorted(0) > sorted(2))
        swap(0, 2);
    if (sorted(1) > sorted(2))
        swap(1, 2);
    return sorted;
}

int vtk2surf(const MatrixXi& tets, MatrixXi& surf)
{
    surf.setZero();
    // assume the tets is tets
    auto comp = [](const Vector3i& lhs, const Vector3i& rhs) -> bool {
        auto sorted_lhs = sort_v(lhs);
        auto sorted_rhs = sort_v(rhs);

        for (size_t i = 0; i < 3; ++i)
        {
            if (sorted_lhs(i) < sorted_rhs(i))
                return true;
        }
        return false;
    };
    //bool is false if it has opposite tet
    auto             faces = map<Vector3i, bool, decltype(comp)>(comp);
    vector<Vector3i> surfs_vec;

    auto insert = [&](const size_t& p, const size_t& q, const size_t& r) -> bool {
        Vector3i one_face;
        one_face << p, q, r;
        auto not_has_oppo = faces.insert({ one_face, true });
        if (!not_has_oppo.second)
            not_has_oppo.first->second = false;
    };

    for (size_t i = 0; i < tets.cols(); ++i)
    {
        insert(0, 1, 2);
        insert(1, 0, 3);
        insert(0, 2, 3);
        insert(2, 1, 3);
    }

    for (auto f_iter = faces.begin(); f_iter != faces.end(); ++f_iter)
    {
        if (f_iter->second)
            surfs_vec.push_back(f_iter->first);
    }
    surf.resize(3, surf.size());

    for (size_t i = 0; i < surf.size(); ++i)
    {
        surf.col(i) = surfs_vec[i];
    }
    return 0;
}

}  //namespace PhysIKA
#endif
