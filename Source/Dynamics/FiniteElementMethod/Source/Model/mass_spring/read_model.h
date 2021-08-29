/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: read model for mass spring method.
 * @version    : 1.0
 */
#ifndef READ_MODEL_JJ_H
#define READ_MODEL_JJ_H

#include "error_deal.h"
#include "copy_macro.h"
#include <cstdio>
#include <Eigen/Core>
#include <string>
#include <vector>
#include <tuple>
#include <array>
#include <iostream>

namespace PhysIKA {
//! \brief read obj from path
//! \param [in] path
//! \return tuple, is_read_success, vertices, cells
template <typename T, typename U = int>
std::tuple<bool, Eigen::Matrix<T, -1, -1>, Eigen::Matrix<U, -1, -1>> ReadObj(const char* path)
{
    FILE* f_in = fopen(path, "r");
    if (!f_in || std::string(path).find(".obj") == std::string::npos)
    {
        printf("[  \033[1;31merror\033[0m  ] error in file open\n");
        return { false, Eigen::Matrix<T, -1, -1>(0, 0), Eigen::Matrix<U, -1, -1>(0, 0) };
    }
    std::vector<std::array<T, 3>> points;
    std::vector<std::array<U, 3>> faces;

    char line[512];
    while (!feof(f_in))
    {
        int scan_success = fscanf(f_in, "%[^\n]%*[\n]", line);
        if (!scan_success)
        {
            printf("[  \033[1;31merror\033[0m  ] error in scan file line\n");
            return { false, Eigen::Matrix<T, -1, -1>(0, 0), Eigen::Matrix<U, -1, -1>(0, 0) };
        }
        if (line[0] == 'v' && line[1] == ' ')
        {
            std::array<T, 3> v = { 0, 0, 0 };
            if (std::is_same<T, double>::value)
                sscanf(line, "%*c%lf%lf%lf", &v[0], &v[1], &v[2]);
            else
                sscanf(line, "%*c%lf%lf%lf", &v[0], &v[1], &v[2]);

            points.push_back(move(v));
        }
        else if (line[0] == 'f' && line[1] == ' ')
        {
            std::array<U, 3> f = { 0, 0, 0 };
            if (std::string(line).find("/") != std::string::npos)
                if (std::is_same<U, size_t>::value)
                    sscanf(line, "%*c%zu%*s%zu%*s%zu", &f[0], &f[1], &f[2]);
                else
                    sscanf(line, "%*c%d%*s%d%*s%d", &f[0], &f[1], &f[2]);
            else if (std::is_same<U, size_t>::value)
                sscanf(line, "%*c%zu%zu%zu", &f[0], &f[1], &f[2]);
            else
                sscanf(line, "%*c%d%d%d", &f[0], &f[1], &f[2]);

            faces.emplace_back(std::array<U, 3>{ f[0] - 1, f[1] - 1, f[2] - 1 });
        }
    }

    fclose(f_in);
    const int                vert_num = points.size();
    const int                face_num = faces.size();
    Eigen::Matrix<T, -1, -1> verts(vert_num, 3);
    Eigen::Matrix<U, -1, -1> cells(face_num, 3);
    for (int v = 0; v < vert_num; ++v)
        THREE_ELEMENT_COPY(verts LBRACKETS v COMMA, RBRACKETS, points[v][, ]);

    for (int f = 0; f < face_num; ++f)
        THREE_ELEMENT_COPY(cells LBRACKETS f COMMA, RBRACKETS, faces[f][, ]);

    return { true, verts, cells };
}

//! \brief read obj from path
//! \param [in] path
//! \return tuple, is_read_success, vertices, cells
template <int E, typename T, typename U = int>
std::tuple<bool, Eigen::Matrix<T, -1, -1>, Eigen::Matrix<U, -1, -1>> ReadTetHex(const char* path)
{
    assert(E == 4 || E == 6);
    FILE* f_in = fopen(path, "r");
    if (!f_in || std::string(path).find(".vtk") == std::string::npos)
    {
        printf("[  \033[1;31merror\033[0m  ] error in file open\n");
        return { false, Eigen::Matrix<T, -1, -1>(0, 0), Eigen::Matrix<U, -1, -1>(0, 0) };
    }
    Eigen::Matrix<T, -1, -1> verts;
    Eigen::Matrix<U, -1, -1> cells;

    char line[512];
    int  vert_num  = 0;
    int  cell_num  = 0;
    int  cell_type = 0;
    while (!feof(f_in))
    {
        EXITIF(fscanf(f_in, "%[^\n]%*[\n]", line) < 1, "scan file line");

        if (line[0] == 'P' && line[1] == 'O' && line[2] == 'I' && line[3] == 'N' && line[4] == 'T' && line[5] == 'S' && line[6] == ' ')
        {
            sscanf(line, "%*s%d", &vert_num);
            verts.resize(vert_num, 3);
            for (int v = 0; v < vert_num; ++v)
            {
                if (std::is_same<T, double>::value)
                    EXITIF(fscanf(f_in, "%lf%lf%lf", &verts(v, 0), &verts(v, 1), &verts(v, 2)) < 3,
                           "file scan vertices")
                else
                    EXITIF(fscanf(f_in, "%f%f%f", &verts(v, 0), &verts(v, 1), &verts(v, 2)) < 3,
                           "file scan vertices");
            }

            while (!(line[0] == 'C' && line[1] == 'E' && line[2] == 'L'
                     && line[3] == 'L' && line[4] == 'S'))
            {
                EXITIF(fscanf(f_in, "\n%s", line) < 1, "CELLS donot found");
                EXITIF(feof(f_in), "unexpected end of file");
            }
            EXITIF(fscanf(f_in, "%d%*d", &cell_num) < 1, "CELLS format error");
            cells.resize(cell_num, E);
            for (int t = 0; t < cell_num; ++t)
            {
                if (E == 4)
                {
                    if (std::is_same<U, size_t>::value)
                        EXITIF(fscanf(f_in, "%*d%zu%zu%zu%zu", &cells(t, 0), &cells(t, 1), &cells(t, 2), &cells(t, 3)) < 4, "cell format error")
                    else
                        EXITIF(fscanf(f_in, "%*d%d%d%d%d", &cells(t, 0), &cells(t, 1), &cells(t, 2), &cells(t, 3)) < 4, "cell format error")
                }
                else
                {
                    if (std::is_same<U, size_t>::value)
                        EXITIF(fscanf(f_in, "%*d%zu%zu%zu%zu%zu%zu", &cells(t, 0), &cells(t, 1), &cells(t, 2), &cells(t, 3), &cells(t, 4), &cells(t, 5)) < 6,
                               "CELLS format error")
                    else
                        EXITIF(fscanf(f_in, "%*d%d%d%d%d%d%d", &cells(t, 0), &cells(t, 1), &cells(t, 2), &cells(t, 3), &cells(t, 4), &cells(t, 5)) < 6,
                               "CELLS format error");
                }
            }
            while (!(line[0] == 'C' && line[1] == 'E' && line[2] == 'L'
                     && line[3] == 'L' && line[4] == '_' && line[5] == 'T' && line[6] == 'Y'))
            {
                EXITIF(feof(f_in), "unexpected end of file");
                EXITIF(fscanf(f_in, "%s", line) < 1, "cell type scan");
            }
            EXITIF(fscanf(f_in, "%*d%d", &cell_type) < 1, "cell type");
            if (E == 4)
                assert(cell_type == 10);
            else if (E == 6)
                assert(cell_type == 12);
            break;
        }
    }

    return { true, verts, cells };
}

template std::tuple<bool, Eigen::Matrix<float, -1, -1>, Eigen::Matrix<int, -1, -1>> ReadObj(const char* path);

template std::tuple<bool, Eigen::Matrix<double, -1, -1>, Eigen::Matrix<int, -1, -1>> ReadObj(const char* path);

template std::tuple<bool, Eigen::Matrix<float, -1, -1>, Eigen::Matrix<int, -1, -1>> ReadTetHex<4>(const char* path);

template std::tuple<bool, Eigen::Matrix<double, -1, -1>, Eigen::Matrix<int, -1, -1>> ReadTetHex<4>(const char* path);

template std::tuple<bool, Eigen::Matrix<float, -1, -1>, Eigen::Matrix<int, -1, -1>> ReadTetHex<6>(const char* path);

template std::tuple<bool, Eigen::Matrix<double, -1, -1>, Eigen::Matrix<int, -1, -1>> ReadTetHex<6>(const char* path);

template std::tuple<bool, Eigen::Matrix<float, -1, -1>, Eigen::Matrix<size_t, -1, -1>> ReadObj(const char* path);

template std::tuple<bool, Eigen::Matrix<double, -1, -1>, Eigen::Matrix<size_t, -1, -1>> ReadObj(const char* path);

template std::tuple<bool, Eigen::Matrix<float, -1, -1>, Eigen::Matrix<size_t, -1, -1>> ReadTetHex<4>(const char* path);

template std::tuple<bool, Eigen::Matrix<double, -1, -1>, Eigen::Matrix<size_t, -1, -1>> ReadTetHex<4>(const char* path);

template std::tuple<bool, Eigen::Matrix<float, -1, -1>, Eigen::Matrix<size_t, -1, -1>> ReadTetHex<6>(const char* path);

template std::tuple<bool, Eigen::Matrix<double, -1, -1>, Eigen::Matrix<size_t, -1, -1>> ReadTetHex<6>(const char* path);

}  // namespace PhysIKA

#endif  // READ_MODEL_JJ_H
