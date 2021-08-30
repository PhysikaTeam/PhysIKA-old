#include "../inc/voxel.h"
const size_t f[6][4] = { { 0, 1, 2, 3 }, { 0, 1, 4, 5 }, { 0, 2, 4, 6 }, { 1, 3, 5, 7 }, { 2, 3, 6, 7 }, { 4, 5, 6, 7 } };
void         corresponding_face_index(const int flag1, const int flag2, const int flag3, std::vector<size_t>& face_index1, std::vector<size_t>& face_index2);
void         corresponding_point_index(const int flag1, const int flag2, const int flag3, std::vector<size_t>& point_index1, std::vector<size_t>& point_index2);
using namespace cxz;

SimpleVoxel::SimpleVoxel(const size_t (&input_index)[3], const size_t& index)
{
    coord_index[0] = input_index[0];
    coord_index[1] = input_index[1];
    coord_index[2] = input_index[2];
    for (size_t i = 0; i < 8; i++)
    {
        vertice_index[i] = -1;
    }
    cell_index   = index;
    if_cell_cube = 0;
}

void SimpleVoxel::voxel_write_to_file(std::ofstream& file)
{
    file << "8 ";
    for (size_t i = 0; i < 8; i++)
        file << vertice_index[i] << " ";
    file << std::endl;
}

void SimpleVoxel::tet_write_to_file(std::ofstream& file, const std::vector<cxz::MyVector3>& vertice)
{
    file << "4 "
         << " " << vertice_index[0] << " " << vertice_index[1] << " " << vertice_index[5] << " " << vertice_index[7] << std::endl;
    file << "4 "
         << " " << vertice_index[0] << " " << vertice_index[1] << " " << vertice_index[3] << " " << vertice_index[7] << std::endl;
    file << "4 "
         << " " << vertice_index[0] << " " << vertice_index[4] << " " << vertice_index[5] << " " << vertice_index[7] << std::endl;
    file << "4 "
         << " " << vertice_index[0] << " " << vertice_index[4] << " " << vertice_index[6] << " " << vertice_index[7] << std::endl;
    file << "4 "
         << " " << vertice_index[0] << " " << vertice_index[2] << " " << vertice_index[3] << " " << vertice_index[7] << std::endl;
    file << "4 "
         << " " << vertice_index[0] << " " << vertice_index[2] << " " << vertice_index[6] << " " << vertice_index[7] << std::endl;
}

void VoxelMesh::coord_to_index(const cxz::MyVector3& coord, size_t (&index)[3])
{
    cxz::MyVector3 temp = coord - relative_coord_origin;
    index[0]            = int(temp.coord[0] / normal_voxel.coord[0]);
    index[1]            = int(temp.coord[1] / normal_voxel.coord[1]);
    index[2]            = int(temp.coord[2] / normal_voxel.coord[2]);
}

cxz::MyVector3 VoxelMesh::index_to_coord(const size_t (&coord_index)[3], const size_t& point_index)
{
    size_t temp;
    size_t flag[3];
    temp = point_index;

    flag[2] = temp / 4;
    temp    = point_index % 4;
    flag[1] = temp / 2;
    flag[0] = temp % 2;

    cxz::MyVector3 my_vertex;
    my_vertex.coord[0] = relative_coord_origin.coord[0] + normal_voxel.coord[0] * (coord_index[0] + flag[0]);
    my_vertex.coord[1] = relative_coord_origin.coord[1] + normal_voxel.coord[1] * (coord_index[1] + flag[1]);
    my_vertex.coord[2] = relative_coord_origin.coord[2] + normal_voxel.coord[2] * (coord_index[2] + flag[2]);

    return my_vertex;
}

bool VoxelMesh::judge_nearby(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2, const SimpleVtk& vtk)
{
    if (compute_index_distance(voxel_1, voxel_2) == 1)
    {
        if (voxel_1.if_cell_cube == 1 && voxel_2.if_cell_cube == 1)
            return 1;
        else
            return judge_nearby_cell(voxel_1, voxel_2, vtk);
    }
    return 0;
}

size_t VoxelMesh::compute_index_distance(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2)
{
    return (abs(int(voxel_1.coord_index[0]) - int(voxel_2.coord_index[0])) + abs(int(voxel_1.coord_index[1]) - int(voxel_2.coord_index[1])) + abs(int(voxel_1.coord_index[2]) - int(voxel_2.coord_index[2])));
}

bool VoxelMesh::judge_nearby_cell(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2, const SimpleVtk& vtk)
{
    SimpleCell cell_1 = vtk.cells[voxel_1.cell_index];
    SimpleCell cell_2 = vtk.cells[voxel_2.cell_index];
    for (auto i : cell_1.faces)
    {
        for (auto j : cell_2.faces)
        {
            if (judge_same_face(i, j))
                return 1;
        }
    }
    return 0;
}

bool VoxelMesh::judge_same_face(const SimpleFace& face_1, const SimpleFace& face_2)
{
    if (face_1.vertex_number != face_2.vertex_number)
        return 0;
    for (auto i : face_1.vertice_index)
    {
        bool flag = 0;
        for (auto j : face_2.vertice_index)
        {
            if (i == j)
            {
                flag = 1;
                break;
            }
        }
        if (flag == 0)
            return 0;
    }
    return 1;
}

bool VoxelMesh::judge_corresponding_vertex_index(const size_t& voxel_1, const size_t& voxel_2, std::vector<size_t>& index1, std::vector<size_t>& index2)
{
    index1.clear();
    index2.clear();

    std::vector<size_t> face_index1, face_index2;
    std::vector<size_t> point_index1, point_index2;

    int flag[3];
    flag[0] = int(voxel[voxel_2].coord_index[0]) - int(voxel[voxel_1].coord_index[0]);
    flag[1] = int(voxel[voxel_2].coord_index[1]) - int(voxel[voxel_1].coord_index[1]);
    flag[2] = int(voxel[voxel_2].coord_index[2]) - int(voxel[voxel_1].coord_index[2]);

    corresponding_face_index(flag[0], flag[1], flag[2], face_index1, face_index2);
    corresponding_point_index(flag[0], flag[1], flag[2], point_index1, point_index2);

    index1 = point_index1;
    index2 = point_index2;
    return 1;
}

void corresponding_face_index(const int flag1, const int flag2, const int flag3, std::vector<size_t>& face_index1, std::vector<size_t>& face_index2)
{
    face_index1.clear();
    face_index2.clear();

    if (flag1 == -1)
    {
        face_index1.push_back(2);
        face_index2.push_back(3);
    }
    if (flag1 == 1)
    {
        face_index1.push_back(3);
        face_index2.push_back(2);
    }

    if (flag2 == -1)
    {
        face_index1.push_back(1);
        face_index2.push_back(4);
    }
    if (flag2 == 1)
    {
        face_index1.push_back(4);
        face_index2.push_back(1);
    }

    if (flag3 == -1)
    {
        face_index1.push_back(0);
        face_index2.push_back(5);
    }
    if (flag3 == 1)
    {
        face_index1.push_back(5);
        face_index2.push_back(0);
    }
}

void corresponding_face_point_index(const std::vector<size_t>& face_index, std::vector<size_t>& point_index)
{
    size_t count[4] = { 1, 1, 1, 1 };
    size_t flag     = face_index.size();
    point_index.clear();
    for (size_t i = 1; i < flag; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            if (f[face_index[0]][j] == f[face_index[i]][0] || f[face_index[0]][j] == f[face_index[i]][1] || f[face_index[0]][j] == f[face_index[i]][2] || f[face_index[0]][j] == f[face_index[i]][3])
                count[j]++;
        }
    }
    for (size_t i = 0; i < 4; i++)
    {
        if (count[i] == flag)
        {
            point_index.push_back(f[face_index[0]][i]);
        }
    }
}

void corresponding_point_index(const int flag1, const int flag2, const int flag3, std::vector<size_t>& point_index1, std::vector<size_t>& point_index2)
{
    std::vector<size_t> face_index1, face_index2;
    corresponding_face_index(flag1, flag2, flag3, face_index1, face_index2);
    corresponding_face_point_index(face_index1, point_index1);
    corresponding_face_point_index(face_index2, point_index2);
}
