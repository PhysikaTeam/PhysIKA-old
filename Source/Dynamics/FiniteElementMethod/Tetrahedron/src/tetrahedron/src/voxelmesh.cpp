#include <fstream>
#include <algorithm>
#include "../inc/voxel.h"
using namespace std;
using namespace cxz;
const size_t NEARBY_POINT[8][3] = { { 1, 2, 4 }, { 0, 3, 5 }, { 0, 3, 6 }, { 1, 2, 7 }, { 0, 5, 6 }, { 1, 4, 7 }, { 2, 4, 7 }, { 3, 5, 6 } };

VoxelMesh::VoxelMesh(SimpleVtk& vtk)
{
    init_parameter(vtk);
    volume              = normal_voxel.coord[0] * normal_voxel.coord[1] * normal_voxel.coord[2];
    voxel_number        = 0;
    flag_vertice_number = 0;
    vertice_number      = 0;
    voxel.clear();
    vertice.clear();
    vertice_voxel.clear();
    std::cout << "Initialize the parameter!" << std::endl;
}

void VoxelMesh::init_voxel_mesh(SimpleVtk& vtk, std::vector<size_t>& part)
{
    //init voxels
    init_vtk_to_voxel(vtk, part);
    //research nearby voxels
    init_adjacent_voxels(vtk);
    std::cout << "Initialize the voxel!" << std::endl;
    //glue voxels as hexs
    std::vector<size_t> order_list;
    generate_order(order_list);
    init_vertice_coord(order_list);
    std::cout << "Initialize the hex!" << std::endl;
}

int VoxelMesh::init_parameter(const SimpleVtk& vtk)
{
    bool   flag = 1;
    double min[3];

    for (auto i : vtk.cells)
    {
        if (i.if_normal == 1)
        {
            flag              = 0;
            size_t max_index  = 0;
            double max_length = 0;
            for (auto j = 1; j < 8; j++)
            {
                if (length(vtk.vertice[i.vertex_index[j]].vertex - vtk.vertice[i.vertex_index[0]].vertex) > max_length)
                {
                    max_length = length(vtk.vertice[i.vertex_index[j]].vertex - vtk.vertice[i.vertex_index[0]].vertex);
                    max_index  = j;
                }
            }
            normal_voxel          = vtk.vertice[i.vertex_index[max_index]].vertex - vtk.vertice[i.vertex_index[0]].vertex;
            normal_voxel.coord[0] = fabs(normal_voxel.coord[0]);
            normal_voxel.coord[1] = fabs(normal_voxel.coord[1]);
            normal_voxel.coord[2] = fabs(normal_voxel.coord[2]);
            min[0] = relative_coord_origin.coord[0] = vtk.vertice[i.vertex_index[0]].vertex.coord[0];
            min[1] = relative_coord_origin.coord[1] = vtk.vertice[i.vertex_index[0]].vertex.coord[1];
            min[2] = relative_coord_origin.coord[2] = vtk.vertice[i.vertex_index[0]].vertex.coord[2];
            break;
        }
    }
    for (auto i : vtk.vertice)
    {
        for (size_t j = 0; j < 3; j++)
        {
            if (i.vertex.coord[j] < min[j])
            {
                min[j] = i.vertex.coord[j];
            }
        }
    }
    for (size_t i = 0; i < 3; i++)
    {
        relative_coord_origin.coord[i] = relative_coord_origin.coord[i] - (2 + int((relative_coord_origin.coord[i] - min[i]) / normal_voxel.coord[i])) * normal_voxel.coord[i];
    }
    return flag;
}

void VoxelMesh::init_vtk_to_voxel(const SimpleVtk& vtk, std::vector<size_t>& part)
{
    size_t temp_index[3] = { 0, 0, 0 };

    for (size_t i = 0; i < vtk.cells_number; i++)
    {
        cxz::MyVector3 temp_coord = vtk.vertice[vtk.cells[i].vertex_index[0]].vertex;
        for (size_t j = 1; j < vtk.cells[i].vertex_number; j++)
        {
            temp_coord = temp_coord + vtk.vertice[vtk.cells[i].vertex_index[j]].vertex;
        }
        temp_coord = 1.0 / vtk.cells[i].vertex_number * temp_coord;
        coord_to_index(temp_coord, temp_index);
        SimpleVoxel v(temp_index, i);
        v.if_cell_cube = vtk.cells[i].if_normal;

        if ((vtk.cells[i].volume > VOXEL_VOLUME_ZERO * volume))
        {
            voxel.push_back(v);
            voxel_number++;
        }
        else
        {
            part.push_back(i);
        }
    }
}

void VoxelMesh::init_adjacent_voxels(const SimpleVtk& vtk)
{
    size_t i, j;
    for (i = 0; i < voxel_number; i++)
    {
        for (j = 0; j < voxel_number; j++)
        {
            if (judge_nearby(voxel[i], voxel[j], vtk) == 1)
                voxel[i].adjacent_voxels.push_back(j);
        }
    }
}

void VoxelMesh::generate_order(std::vector<size_t>& order_list)
{
    std::vector<int>      origin_order, new_order;
    vector<int>::iterator ret;
    size_t                temp_index = 0, origin_number = 0, count = 0;

    for (size_t i = 0; i < voxel.size(); i++)
        origin_order.push_back(i);
    origin_number = origin_order.size();
    while (origin_number > new_order.size())
    {
        if (temp_index == new_order.size())
        {
            count++;
            for (size_t i = 0; i < origin_number; i++)
            {
                ret = std::find(new_order.begin(), new_order.end(), i);
                if (ret == new_order.end())
                {
                    new_order.push_back(origin_order[i]);
                    break;
                }
            }
        }
        else
        {
            while (temp_index < new_order.size())
            {
                for (auto j : voxel[new_order[temp_index]].adjacent_voxels)
                {
                    ret = std::find(new_order.begin(), new_order.end(), j);
                    if (ret == new_order.end())
                    {
                        new_order.push_back(j);
                    }
                }
                temp_index++;
            }
        }
    }
    order_list.clear();
    for (auto i : new_order)
        order_list.push_back(i);
    if (count == 1)
        std::cout << "--- Check: All the hexs are connected! ---" << std::endl;
    else
        std::cout << "--- Check: All the hexs are not connected! ---" << std::endl;
}

void VoxelMesh::init_vertice_coord(const std::vector<size_t>& order_list)
{
    std::vector<size_t> index1, index2;
    size_t              count = 0;
    bool                flag;

    for (auto i : order_list)
    {
        count++;
        for (auto j : voxel[i].adjacent_voxels)
        {
            if (judge_corresponding_vertex_index(i, j, index1, index2))
            {
                for (size_t k = 0; k < index1.size(); k++)
                {
                    if (voxel[i].vertice_index[index1[k]] == size_t(-1) && voxel[j].vertice_index[index2[k]] != size_t(-1))
                    {
                        voxel[i].vertice_index[index1[k]] = voxel[j].vertice_index[index2[k]];
                        vertice_voxel[voxel[j].vertice_index[index2[k]]].push_back(i);
                    }
                }
            }
            flag = 1;
            for (size_t ii = 0; ii < 8; ii++)
            {
                if (voxel[i].vertice_index[ii] == size_t(-1))
                    flag = 0;
            }
            if (flag == 1)
                break;
        }
        if (flag == 0)
        {
            for (size_t jj = 0; jj < 8; jj++)
            {
                if (voxel[i].vertice_index[jj] == size_t(-1))
                {
                    voxel[i].vertice_index[jj] = vertice_number;
                    vertice.push_back(index_to_coord(voxel[i].coord_index, jj));
                    std::vector<size_t> temp1, temp2;
                    temp1.push_back(i);
                    vertice_voxel.push_back(temp1);
                    temp2.push_back(jj);
                    vertice_number++;
                }
            }
        }
    }
    flag_vertice_number = vertice_number;
}

void VoxelMesh::write_hex_to_vtk_file(const char* const path)
{
    ofstream file;
    file.open(path);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "Unstructured Grid Example" << endl;
    file << "ASCII" << endl;
    file << "DATASET UNSTRUCTURED_GRID" << endl;

    file << "POINTS " << flag_vertice_number << " double" << endl;
    for (size_t i = 0; i < flag_vertice_number; i++)
        file << vertice[i].coord[0] << " " << vertice[i].coord[1] << " " << vertice[i].coord[2] << endl;

    file << "CELLS " << voxel_number << " " << voxel_number * 9 << endl;
    for (auto i : voxel)
    {
        i.voxel_write_to_file(file);
    }
    file << "CELL_TYPES " << voxel_number << endl;
    for (auto i = 0; i < voxel_number; i++)
    {
        file << 11 << endl;
    }
    file.close();
    std::cout << "Write hex to file!" << std::endl;
}

void VoxelMesh::write_tet_to_vtk_file(const char* const path)
{
    ofstream file;
    file.open(path);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "Unstructured Grid Example" << endl;
    file << "ASCII" << endl;
    file << "DATASET UNSTRUCTURED_GRID" << endl;

    file << "POINTS " << flag_vertice_number << " double" << endl;
    for (size_t i = 0; i < flag_vertice_number; i++)
        file << vertice[i].coord[0] << " " << vertice[i].coord[1] << " " << vertice[i].coord[2] << endl;

    size_t tet_number = 6 * voxel_number;
    file << "CELLS " << tet_number << " " << tet_number * 5 << endl;
    for (auto i : voxel)
    {
        i.tet_write_to_file(file, vertice);
    }
    file << "CELL_TYPES " << tet_number << endl;
    for (auto i = 0; i < tet_number; i++)
    {
        file << 10 << endl;
    }
    file.close();
    std::cout << "Write tet to file!" << std::endl;
}