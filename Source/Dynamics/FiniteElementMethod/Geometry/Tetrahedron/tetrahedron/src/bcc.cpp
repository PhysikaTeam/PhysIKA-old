#include <fstream>
#include "../inc/voxel.h"
using namespace std;
using namespace cxz;

const size_t bcc_orientation[6][4] = { { 0, 2, 6, 4 }, { 1, 3, 7, 5 }, { 0, 1, 5, 4 }, { 2, 3, 7, 6 }, { 0, 1, 3, 2 }, { 4, 5, 7, 6 } };

void VoxelMesh::bcc_generate_tet(const SimpleVtk& vtk)
{
    insert_bcc_center();
    init_bcc_segment();
    init_bcc_tet();
    init_true_bcc_tet(vtk);
    std::cout << "Initlize the bcc!" << std::endl;
}

size_t VoxelMesh::orientation_between_voxel(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2)
{
    int flag[3];
    for (size_t i = 0; i < 3; i++)
        flag[i] = int(voxel_2.coord_index[i]) - int(voxel_1.coord_index[i]);
    if (flag[0] == -1)
        return 0;
    if (flag[0] == 1)
        return 1;
    if (flag[1] == -1)
        return 2;
    if (flag[1] == 1)
        return 3;
    if (flag[2] == -1)
        return 4;
    if (flag[2] == 1)
        return 5;
    return -1;
}

void VoxelMesh::insert_bcc_center()
{
    for (size_t i = 0; i < voxel_number; i++)
    {
        MyVector3 temp_coord = index_to_coord(voxel[i].coord_index, 0);
        temp_coord           = temp_coord + 0.5 * normal_voxel;
        vertice.push_back(temp_coord);
        std::vector<size_t> temp_index;
        vertice_voxel.push_back(temp_index);
    }
    vertice_number = vertice.size();
}

void VoxelMesh::init_bcc_segment()
{
    for (size_t i = 0; i < voxel_number; i++)
    {
        vector<size_t> temp_index(3);
        bool           flag[6];
        for (size_t j = 0; j < 6; j++)
            flag[j] = 0;
        temp_index[0] = i;
        for (auto j : voxel[i].adjacent_voxels)
        {
            temp_index[1]       = j;
            temp_index[2]       = orientation_between_voxel(voxel[i], voxel[j]);
            flag[temp_index[2]] = 1;
            if (i < j)
                bcc_segment.push_back(temp_index);
        }
        for (size_t j = 0; j < 6; j++)
        {
            if (flag[j] == 0)
            {
                temp_index[1] = i;
                temp_index[2] = j;
                bcc_segment.push_back(temp_index);
            }
        }
    }
}

void VoxelMesh::init_bcc_tet()
{
    for (size_t i = 0; i < bcc_segment.size(); i++)
    {
        if (bcc_segment[i][0] != bcc_segment[i][1])
        {
            vector<size_t> temp_index(4);
            temp_index[0] = flag_vertice_number + bcc_segment[i][0];
            temp_index[1] = flag_vertice_number + bcc_segment[i][1];
            for (size_t j = 0; j < 4; j++)
            {
                size_t temp1  = bcc_orientation[bcc_segment[i][2]][j];
                size_t temp2  = bcc_orientation[bcc_segment[i][2]][(j + 1) % 4];
                temp_index[2] = voxel[bcc_segment[i][0]].vertice_index[temp1];
                temp_index[3] = voxel[bcc_segment[i][0]].vertice_index[temp2];
                bcc_tet.push_back(temp_index);
            }
        }
        else
        {
            vector<size_t> temp_index(4);
            temp_index[0] = flag_vertice_number + bcc_segment[i][0];
            temp_index[1] = vertice_number;

            MyVector3 temp_coord;
            for (size_t j = 0; j < 4; j++)
            {
                size_t temp_ = bcc_orientation[bcc_segment[i][2]][j];
                temp_coord   = temp_coord + vertice[voxel[bcc_segment[i][0]].vertice_index[temp_]];
            }
            vertice.push_back(0.25 * temp_coord);
            vertice_number++;
            std::vector<size_t> my_temp_index;
            vertice_voxel.push_back(my_temp_index);

            for (size_t j = 0; j < 4; j++)
            {
                size_t temp1  = bcc_orientation[bcc_segment[i][2]][j];
                size_t temp2  = bcc_orientation[bcc_segment[i][2]][(j + 1) % 4];
                temp_index[2] = voxel[bcc_segment[i][0]].vertice_index[temp1];
                temp_index[3] = voxel[bcc_segment[i][0]].vertice_index[temp2];
                bcc_tet.push_back(temp_index);
            }
        }
    }
}

void VoxelMesh::init_true_bcc_tet(const SimpleVtk& vtk)
{
    for (size_t i = 0; i < vertice_number; i++)
    {
        vertice_voxel[i].clear();
    }

    size_t count = 0;
    for (size_t i = 0; i < bcc_tet.size(); i++)
    {
        size_t index1 = bcc_segment[i / 4][0];
        size_t index2 = bcc_segment[i / 4][1];
        if (voxel[index1].if_cell_cube == 1 && voxel[index2].if_cell_cube == 1)
        {
            true_bcc_tet.push_back(i);
            for (size_t j = 0; j < 4; j++)
            {
                vertice_voxel[bcc_tet[i][j]].push_back(i);
            }
        }
        else
        {
            if (index1 == index2)
            {
                if (judge_tet_cell_intersect(i, voxel[index1].cell_index, vtk))
                {
                    true_bcc_tet.push_back(i);
                    for (size_t j = 0; j < 4; j++)
                    {
                        vertice_voxel[bcc_tet[i][j]].push_back(i);
                    }
                }
            }
            else
            {
                if (judge_tet_cell_intersect(i, voxel[index1].cell_index, vtk) || judge_tet_cell_intersect(i, voxel[index2].cell_index, vtk))
                {
                    true_bcc_tet.push_back(i);
                    for (size_t j = 0; j < 4; j++)
                    {
                        vertice_voxel[bcc_tet[i][j]].push_back(i);
                    }
                }
            }
        }
    }
}

bool VoxelMesh::judge_tet_cell_intersect(const size_t& tet_index, const size_t& cell_index, const SimpleVtk& vtk)
{
    MyVector3 my_tet[4];
    MyVector3 center_coord;
    for (size_t j = 0; j < 4; j++)
    {
        my_tet[j]    = vertice[bcc_tet[tet_index][j]];
        center_coord = center_coord + my_tet[j];
    }
    center_coord = 0.25 * center_coord;

    SimpleCell my_cell = vtk.cells[cell_index];
    MyVector3  my_cell_tet[4];
    bool       my_flag = 0;
    for (auto i : my_cell.faces)
    {
        my_flag = 0;
        for (auto j : i.vertice_index)
        {
            if (j == my_cell.vertex_index[0])
            {
                my_flag = 1;
            }
        }
        if (my_flag == 0)
        {
            for (size_t j = 1; j < i.vertex_number - 1; j++)
            {
                my_cell_tet[0] = vtk.vertice[my_cell.vertex_index[0]].vertex;
                my_cell_tet[1] = vtk.vertice[i.vertice_index[0]].vertex;
                my_cell_tet[2] = vtk.vertice[i.vertice_index[j]].vertex;
                my_cell_tet[3] = vtk.vertice[i.vertice_index[j + 1]].vertex;
                if (point_in_tet(center_coord, my_cell_tet))
                    return 1;
            }
        }
    }
    return 0;
}

void VoxelMesh::write_bcc_tet_to_vtk_file(const char* const path)
{
    ofstream file;
    file.open(path);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "Unstructured Grid Example" << endl;
    file << "ASCII" << endl;
    file << "DATASET UNSTRUCTURED_GRID" << endl;

    size_t              real_vertice_number = 0;
    std::vector<size_t> real_vertice_index(vertice_number);
    for (size_t i = 0; i < vertice_number; i++)
    {
        if (vertice_voxel[i].size() > 0)
        {
            real_vertice_index[i] = real_vertice_number;
            real_vertice_number++;
        }
        else
        {
            real_vertice_index[i] = size_t(-1);
        }
    }

    file << "POINTS " << real_vertice_number << " double" << endl;
    for (size_t i = 0; i < vertice_number; i++)
    {
        if (real_vertice_index[i] != size_t(-1))
        {
            file << vertice[i].coord[0] << " " << vertice[i].coord[1] << " " << vertice[i].coord[2] << endl;
        }
    }
    size_t tet_number = true_bcc_tet.size();
    file << "CELLS " << tet_number << " " << tet_number * 5 << endl;
    for (auto i : true_bcc_tet)
    {
        file << "4 "
             << " " << real_vertice_index[bcc_tet[i][0]] << " " << real_vertice_index[bcc_tet[i][1]] << " " << real_vertice_index[bcc_tet[i][2]] << " " << real_vertice_index[bcc_tet[i][3]] << std::endl;
    }
    file << "CELL_TYPES " << tet_number << endl;
    for (auto i = 0; i < tet_number; i++)
    {
        file << 10 << endl;
    }
    file.close();
    std::cout << "Write bcc tet to file!" << std::endl;
}

void VoxelMesh::write_all_bcc_tet_to_vtk_file(const char* const path)
{
    ofstream file;
    file.open(path);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "Unstructured Grid Example" << endl;
    file << "ASCII" << endl;
    file << "DATASET UNSTRUCTURED_GRID" << endl;

    file << "POINTS " << vertice_number << " double" << endl;
    for (auto i : vertice)
        file << i.coord[0] << " " << i.coord[1] << " " << i.coord[2] << endl;
    size_t tet_number = bcc_tet.size();
    file << "CELLS " << tet_number << " " << tet_number * 5 << endl;
    for (size_t i = 0; i < tet_number; i++)
    {
        file << "4 "
             << " " << bcc_tet[i][0] << " " << bcc_tet[i][1] << " " << bcc_tet[i][2] << " " << bcc_tet[i][3] << std::endl;
    }
    file << "CELL_TYPES " << tet_number << endl;
    for (auto i = 0; i < tet_number; i++)
    {
        file << 10 << endl;
    }
    file.close();
    std::cout << "Write all bcc tet to file!" << std::endl;
}