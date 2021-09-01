/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: vtk format io utility
 * @version    : 1.0
 */
#ifndef VTK_H
#define VTK_H

template <typename OS, typename FLOAT, typename INT>
void line2vtk(
    OS&          os,
    const FLOAT* node,
    size_t       node_num,
    const INT*   line,
    size_t       line_num)
{
    os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << node_num << " float\n";
    for (size_t i = 0; i < node_num; ++i)
        os << node[i * 3 + 0] << " " << node[i * 3 + 1] << " " << node[i * 3 + 2] << "\n";

    os << "CELLS " << line_num << " " << line_num * 3 << "\n";
    for (size_t i = 0; i < line_num; ++i)
        os << 2 << " " << line[i * 2 + 0] << " " << line[i * 2 + 1] << "\n";

    os << "CELL_TYPES " << line_num << "\n";
    for (size_t i = 0; i < line_num; ++i)
        os << 3 << "\n";
}

template <typename OS, typename FLOAT, typename INT>
void point2vtk(OS&          os,
               const FLOAT* node,
               size_t       node_num,
               const INT*   points,
               size_t       points_num)
{
    os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << node_num << " float\n";
    for (size_t i = 0; i < node_num; ++i)
        os << node[i * 3 + 0] << " " << node[i * 3 + 1] << " " << node[i * 3 + 2] << "\n";

    os << "CELLS " << points_num << " " << points_num * 2 << "\n";
    for (size_t i = 0; i < points_num; ++i)
        os << 1 << " " << points[i] << "\n";

    os << "CELL_TYPES " << points_num << "\n";
    for (size_t i = 0; i < points_num; ++i)
        os << 1 << "\n";
}

template <typename OS, typename FLOAT, typename INT>
void tri2vtk(
    OS&          os,
    const FLOAT* node,
    size_t       node_num,
    const INT*   tri,
    size_t       tri_num)
{
    os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << node_num << " float\n";
    for (size_t i = 0; i < node_num; ++i)
        os << node[i * 3 + 0] << " " << node[i * 3 + 1] << " " << node[i * 3 + 2] << "\n";

    os << "CELLS " << tri_num << " " << tri_num * 4 << "\n";
    for (size_t i = 0; i < tri_num; ++i)
        os << 3 << "  " << tri[i * 3 + 0] << " " << tri[i * 3 + 1] << " " << tri[i * 3 + 2] << "\n";
    os << "CELL_TYPES " << tri_num << "\n";
    for (size_t i = 0; i < tri_num; ++i)
        os << 5 << "\n";
}

template <typename OS, typename FLOAT, typename INT>
void quad2vtk(
    OS&          os,
    const FLOAT* node,
    size_t       node_num,
    const INT*   quad,
    size_t       quad_num)
{
    os << "# vtk DataFile Version 2.0\nTRI\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";

    os << "POINTS " << node_num << " float\n";
    for (size_t i = 0; i < node_num; ++i)
        os << node[i * 3 + 0] << " " << node[i * 3 + 1] << " " << node[i * 3 + 2] << "\n";

    os << "CELLS " << quad_num << " " << quad_num * 5 << "\n";
    for (size_t i = 0; i < quad_num; ++i)
        os << 4 << "  " << quad[i * 4 + 0] << " " << quad[i * 4 + 1] << " " << quad[i * 4 + 2] << " " << quad[i * 4 + 3] << "\n";
    os << "CELL_TYPES " << quad_num << "\n";
    for (size_t i = 0; i < quad_num; ++i)
        os << 9 << "\n";
}
template <typename OS, typename FLOAT, typename INT>
void tet2vtk(
    OS&          os,
    const FLOAT* node,
    size_t       node_num,
    const INT*   tet,
    size_t       tet_num)
{
    os << "# vtk DataFile Version 2.0\nTET\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";
    os << "POINTS " << node_num << " float\n";
    for (size_t i = 0; i < node_num; ++i)
        os << node[i * 3 + 0] << " " << node[i * 3 + 1] << " " << node[i * 3 + 2] << "\n";

    os << "CELLS " << tet_num << " " << tet_num * 5 << "\n";
    for (size_t i = 0; i < tet_num; ++i)
        os << 4 << "  "
           << tet[i * 4 + 0] << " " << tet[i * 4 + 1] << " "
           << tet[i * 4 + 2] << " " << tet[i * 4 + 3] << "\n";
    os << "CELL_TYPES " << tet_num << "\n";
    for (size_t i = 0; i < tet_num; ++i)
        os << 10 << "\n";
}
template <typename OS, typename FLOAT, typename INT>
void hex2vtk(
    OS&          os,
    const FLOAT* node,
    size_t       node_num,
    const INT*   hex,
    size_t       hex_num)
{
    os << "# vtk DataFile Version 2.0\nTET\nASCII\n\nDATASET UNSTRUCTURED_GRID\n";
    os << "POINTS " << node_num << " float\n";
    for (size_t i = 0; i < node_num; ++i)
        os << node[i * 3 + 0] << " " << node[i * 3 + 1] << " " << node[i * 3 + 2] << "\n";

    os << "CELLS " << hex_num << " " << hex_num * 9 << "\n";
    // for(size_t i = 0; i < hex_num; ++i)
    //   os << 8 << "  "
    //      << hex[i*8+7] << " " << hex[i*8+5] << " "
    //      << hex[i*8+4] << " " << hex[i*8+6] << " "
    //      << hex[i*8+3] << " " << hex[i*8+1] << " "
    //      << hex[i*8+0] << " " << hex[i*8+2] << "\n";

    for (size_t i = 0; i < hex_num; ++i)
    {
        os << 8 << " ";
        for (size_t j = 0; j < 8; ++j)
        {
            os << hex[i * 8 + j] << " ";
        }
        os << "\n";
    }

    os << "CELL_TYPES " << hex_num << "\n";
    for (size_t i = 0; i < hex_num; ++i)
        os << 12 << "\n";
}

template <typename OS, typename Iterator, typename INT>
void vtk_data(OS& os, Iterator first, INT size, const char* value_name, const char* table_name = "my_table")
{
    os << "SCALARS " << value_name << " float\nLOOKUP_TABLE " << table_name << "\n";
    for (size_t i = 0; i < size; ++i, ++first)
        os << *first << "\n";
}

template <typename OS, typename Iterator, typename INT>
void vtk_vector(OS& os, Iterator first, INT size, const char* vector_name)
{
    os << "VECTORS " << vector_name << " double\n";
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < 3; ++j, ++first)
            os << *first << " ";
        os << "\n";
    }
}

template <typename OS, typename Iterator, typename INT>
void vtk_data_rgba(OS& os, Iterator first, INT size, const char* value_name, const char* table_name = "my_table")
{
    os << "COLOR_SCALARS " << value_name << " 4\n";  //\nLOOKUP_TABLE " << table_name << "\n";
    for (size_t i = 0; i < size; ++i)
    {
        for (size_t j = 0; j < 4; ++j, ++first)
        {
            if (j != 3)
                os << *first << " ";
            else
                os << *first;
        }
        os << "\n";
    }
}
template <typename OS, typename Iterator, typename INT>
void point_data(OS& os, Iterator first, INT size, const char* value_name, const char* table_name = "my_table")
{
    os << "POINT_DATA " << size << "\n";
    vtk_data(os, first, size, value_name, table_name);
}

template <typename OS, typename Iterator, typename INT>
void cell_data(OS& os, Iterator first, INT size, const char* value_name, const char* table_name = "my_table")
{
    os << "CELL_DATA " << size << "\n";
    vtk_data(os, first, size, value_name, table_name);
}

template <typename OS, typename Iterator, typename INT>
void cell_data_rgba(OS& os, Iterator first, INT size, const char* value_name, const char* table_name = "my_table")
{
    os << "CELL_DATA " << size << "\n";
    vtk_data_rgba(os, first, size, value_name, table_name);
}

template <typename OS, typename Iterator, typename INT>
void point_data_rgba(OS& os, Iterator first, INT size, const char* value_name, const char* table_name = "my_table")
{
    os << "POINT_DATA " << size << "\n";
    vtk_data_rgba(os, first, size, value_name, table_name);
}

template <typename OS, typename Iterator, typename INT>
void cell_data_rgba_and_scalar(OS& os, Iterator rgba_first, Iterator scalar_first, INT size, const char* rgba_value_name, const char* scalar_value_name, const char* table_name = "my_table")
{
    os << "CELL_DATA " << size << "\n";
    vtk_data_rgba(os, rgba_first, size, rgba_value_name, table_name);
    vtk_data(os, scalar_first, size, scalar_value_name, table_name);
}

template <typename OS, typename Iterator, typename INT>
void point_data_vector(bool is_append, OS& os, Iterator first, INT size, const char* vector_name)
{
    if (!is_append)
        os << "POINT_DATA " << size << "\n";
    vtk_vector(os, first, size, vector_name);
}
template <typename OS, typename Iterator, typename INT>
void point_data_scalar(bool is_append, OS& os, Iterator first, INT size, const char* scalar_name)
{
    if (!is_append)
        os << "POINT_DATA " << size << "\n";
    vtk_data(os, first, size, scalar_name);
}

#endif
