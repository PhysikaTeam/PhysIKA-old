#include <fstream>
#include <sstream>
#include "../inc/vtk.h"
#include "../inc/vector.h"
using namespace std;

int SimpleVtk::read_file_vtk42(const char* const path)
{
    ifstream           file(path);
    string             line, temp_string;
    size_t             number, number1, number2, type;
    double             temp_point[3];
    SimpleVertex*      p = nullptr;
    SimpleFace*        f = nullptr;
    SimpleCell*        c = nullptr;
    vector<SimpleFace> temp_face;
    vector<size_t>     temp_vertice;

    while (getline(file, line))
    {
        istringstream in(line);
        in >> temp_string;
        if (temp_string == "POINTS")
        {
            in >> number;
            vertice_number = number;
            for (size_t i = 0; i < vertice_number; i++)
            {
                getline(file, line);
                in.clear();
                in.str(line);
                in >> temp_point[0] >> temp_point[1] >> temp_point[2];
                p = new SimpleVertex(temp_point[0], temp_point[1], temp_point[2]);
                vertice.push_back(*p);
                delete p;
            }
        }

        if (temp_string == "CELLS")
        {
            in >> number;
            cells_number = number;
            for (size_t i = 0; i < cells_number; i++)
            {
                getline(file, line);
                in.clear();
                in.str(line);
                in >> number >> number1;
                temp_face.clear();
                for (size_t j = 0; j < number1; j++)
                {
                    in >> number2;
                    temp_vertice.clear();
                    for (size_t k = 0; k < number2; k++)
                    {
                        in >> number;
                        temp_vertice.push_back(number);
                    }
                    f = new SimpleFace(temp_vertice, vertice);
                    temp_face.push_back(*f);
                    delete f;
                }
                c = new SimpleCell(temp_face);
                cells.push_back(*c);
                delete c;
            }
        }

        if (temp_string == "CELL_TYPES")
        {
            in >> number;
            for (size_t i = 0; i < number; i++)
            {
                getline(file, line);
                in.clear();
                in.str(line);
                in >> type;
                if (type != 42)
                    return 1;
            }
        }
    }

    volume = 0;
    for (size_t i = 0; i < cells_number; i++)
    {
        volume = volume + cells[i].compute_cell_volume(vertice);
    }
    limit_volume = volume / cells_number * VTK_VOLUME_ZERO;

    return 0;
}

int SimpleVtk::write_to_file(const char* const path)
{
    ofstream file;
    file.open(path);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "Unstructured Grid Example" << endl;
    file << "ASCII" << endl;
    file << "DATASET UNSTRUCTURED_GRID" << endl;

    file << "POINTS " << vertice_number << " double" << endl;
    for (auto i : vertice)
        file << i.vertex.coord[0] << " " << i.vertex.coord[1] << " " << i.vertex.coord[2] << endl;

    size_t number = 0;
    for (auto i : cells)
    {
        number = number + i.vtk_format_number();
    }
    file << "CELLS " << cells_number << " " << number << endl;
    for (auto i : cells)
    {
        i.cell_write_to_file(file);
    }
    file << "CELL_TYPES " << cells_number << endl;
    for (auto i = 0; i < cells_number; i++)
    {
        file << 42 << endl;
    }
    file.close();

    return 0;
}

int SimpleVtk::write_to_file_part(const std::vector<size_t>& cell_index, const char* const path)
{
    ofstream file;
    file.open(path);
    file << "# vtk DataFile Version 2.0" << endl;
    file << "Unstructured Grid Example" << endl;
    file << "ASCII" << endl;
    file << "DATASET UNSTRUCTURED_GRID" << endl;

    file << "POINTS " << vertice_number << " double" << endl;
    for (auto i : vertice)
        file << i.vertex.coord[0] << " " << i.vertex.coord[1] << " " << i.vertex.coord[2] << endl;

    size_t number = 0;
    for (auto i : cell_index)
    {
        number = number + cells[i].vtk_format_number();
    }
    file << "CELLS " << cell_index.size() << " " << number << endl;
    for (auto i : cell_index)
    {
        cells[i].cell_write_to_file(file);
    }
    file << "CELL_TYPES " << cell_index.size() << endl;
    for (auto i = 0; i < cell_index.size(); i++)
    {
        file << 42 << endl;
    }
    file.close();

    return 0;
}