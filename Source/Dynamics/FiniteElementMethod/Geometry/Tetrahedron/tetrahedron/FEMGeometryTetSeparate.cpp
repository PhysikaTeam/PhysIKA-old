#include <iostream>
#include "FEMGeometryTetVtk.h"
using namespace std;

int  separate_cell(const SimpleCell& origin_cell, SimpleCell& cell_0, SimpleCell& cell_1);
bool intersect(SimpleCell& cell, const SimpleFace& face);
bool intersect(SimpleCell& cell, const size_t& vertex);

int separate_vtk(const SimpleVtk& origin_vtk, SimpleVtk& new_vtk)
{
    new_vtk.vertice = origin_vtk.vertice;
    for (auto i : origin_vtk.cells)
    {
        SimpleCell new_cell(i.faces);
        bool       flag  = 0;
        size_t     count = 0;
        while (flag == 0)
        {
            SimpleCell cell_0, cell_1;
            flag = separate_cell(new_cell, cell_0, cell_1);
            if (flag == 0 || count != 0)
            {
                count++;
                cell_0.if_normal = 2;
            }
            new_vtk.cells.push_back(cell_0);
            new_cell = cell_1;
        }
    }
    new_vtk.vertice_number = new_vtk.vertice.size();
    new_vtk.cells_number   = new_vtk.cells.size();
    new_vtk.volume         = 0;
    for (size_t i = 0; i < new_vtk.cells_number; i++)
    {
        new_vtk.volume = new_vtk.volume + new_vtk.cells[i].compute_cell_volume(new_vtk.vertice);
    }
    new_vtk.limit_volume = new_vtk.volume / new_vtk.cells_number * VTK_VOLUME_ZERO;
    return 0;
}

int separate_cell(const SimpleCell& origin_cell, SimpleCell& cell_0, SimpleCell& cell_1)
{
    cell_0.clear();
    cell_1.clear();
    bool           flag = 1;
    bool           flag_1;
    size_t         face_number = origin_cell.face_number;
    vector<size_t> face_index;

    if (origin_cell.face_number == 0)
        return 1;
    cell_0.AddFace(origin_cell.faces[0]);
    face_index.push_back(0);
    while (flag != 0)
    {
        flag = 0;
        for (size_t i = 0; i < face_number; i++)
        {
            flag_1 = 0;
            for (auto j : face_index)
            {
                if (i == j)
                    flag_1 = 1;
            }
            if (flag_1 == 0)
            {
                if (intersect(cell_0, origin_cell.faces[i]))
                {
                    cell_0.AddFace(origin_cell.faces[i]);
                    face_index.push_back(i);
                    flag = 1;
                }
            }
        }
    }
    if (cell_0.face_number == origin_cell.face_number)
        return 1;
    else
    {
        for (size_t i = 0; i < face_number; i++)
        {
            flag_1 = 0;
            for (auto j : face_index)
            {
                if (i == j)
                    flag_1 = 1;
            }
            if (flag_1 == 0)
            {
                cell_1.AddFace(origin_cell.faces[i]);
            }
        }
    }
    return 0;
}

bool intersect(SimpleCell& cell, const SimpleFace& face)
{
    if (cell.face_number == 0 || face.vertex_number == 0)
        return 0;
    for (auto i : face.vertice_index)
    {
        if (intersect(cell, i))
            return 1;
    }
    return 0;
}

bool intersect(SimpleCell& cell, const size_t& vertex)
{
    if (cell.face_number == 0)
        return 0;
    for (auto i : cell.vertex_index)
    {
        if (i == vertex)
            return 1;
    }
    return 0;
}