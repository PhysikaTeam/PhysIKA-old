#ifndef CONSTRUCT_MESH_JJ_H
#define CONSTRUCT_MESH_JJ_H

#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <array>


template<typename D = int>
std::unordered_set<array<int, 2>> GetAllEdges(const Eigen::Matrix<D, -1, -1> &cells)
{
  const int R = cells.rows();
  assert(R == 4);
  const int C = cells.cols();
  const std::vector<std::array<int, 2>> edges_idx = {{0, 1}, {1, 2}, {2, 0}, {3, 0}, {3, 1}, {3, 2}};
  std::unordered_set<array<int, 2>> all_edges;
  for (int c = 0; c < C; ++c)
  {
    for (int i = 0; i < 6; ++i)
    {
      std::array<int, 2> edge = {cells(edges_idx[i][0], c),
                                 cells(edges_idx[i][1], c)};
      all_edges.insert(edge);
    }
  }

  return all_edges;
}



template<typename T = float, typename D = int>
void GetMeshEdgeConnection(const Eigen::Matrix<D, -1, -1> &cells, Eigen::SparseMatrix<T> &laplacian,
                           Eigen::SparseMatrix<T> &edge_connect, int &edge_num)
{
  std::unordered_set<array<int, 2>> all_edges = GetAllEdges(cells);
  edge_num = all_edges.size();
  
  std::unordered_map<std::array<int, 2>, T> laplacian_ele;
  std::vector<Eigen::Triplet<T>> connect_ele;
  int edge_count = 0;
  int vert_num = 0;
  for (const auto &e : all_edges)
  {
    vert_num = max(e[0]+1, vert_num);
    vert_num = max(e[1]+1, vert_num);
      
    laplacian_ele[{e[0], e[0]}] = laplacian_ele[{e[0], e[0]}] + 1;
    laplacian_ele[{e[0], e[1]}] = -1;
    laplacian_ele[{e[1], e[0]}] = -1;
    laplacian_ele[{e[1], e[1]}] = laplacian_ele[{e[1], e[1]}] + 1;
      
    connect_ele.emplace_back({e[0], edge_count, 1});
    connect_ele.emplace_back({e[1], edge_count, -1});
    edge_count += 1;

  }

  laplacian = Eigen::SparseMatrix<T>(vert_num, vert_num);
  edge_connect = Eigen::SparseMatrix<T>(vert_num, edge_count);

  laplacian.setFromTriplets(laplacian_ele.begin(), laplacian_ele.end());
  edge_connect.setFromTriplets(connect_ele.begin(), connect_ele.end());

}
template<typename T = float, typename D = int>
Eigen::Matrix<T, -1, 1> GetEdgeOriginLength(const Eigen::Matrix<T, -1, -1> &nods, const Eigen::Matrix<int, -1, -1> &cells)
{
  std::unordered_set<array<int, 2>> all_edges = GetAllEdges(cells);
  const int edge_num = all_edges.size();
  Eigen::Matrix<T, -1, 1> edge_length(edge_num);

  int edge_count = 0;
  for (const auto &e : all_edges)
  {
    edge_length[edge_count++] = (nods.col(e[0]) - nods.col(e[1])).norm();
  }

  return edge_length;
}


    
#endif // CONSTRUCT_MESH_JJ_H
