#include "../inc/vtk.h"
#include "../inc/vector.h"
using namespace std;
SimpleVertex::SimpleVertex(const double &p1,const double &p2,const double &p3)
{
    vertex.coord[0]=p1;
    vertex.coord[1]=p2;
    vertex.coord[2]=p3;
    sign=1;
}

void SimpleVertex::delete_vertex()
{
    sign=0;
}

SimpleFace::SimpleFace(const std::vector<size_t> &input_vertice,const std::vector<SimpleVertex> &vertice)
{
    vertex_number=input_vertice.size();
    vertice_index=input_vertice;

    if(vertex_number==4){
        if((vertice[vertice_index[1]].vertex-vertice[vertice_index[0]].vertex)==(vertice[vertice_index[3]].vertex-vertice[vertice_index[2]].vertex) 
        || (vertice[vertice_index[1]].vertex-vertice[vertice_index[0]].vertex)==(vertice[vertice_index[2]].vertex-vertice[vertice_index[3]].vertex)){
            if((vertice[vertice_index[3]].vertex-vertice[vertice_index[0]].vertex)==(vertice[vertice_index[1]].vertex-vertice[vertice_index[2]].vertex) 
            || (vertice[vertice_index[3]].vertex-vertice[vertice_index[0]].vertex)==(vertice[vertice_index[2]].vertex-vertice[vertice_index[1]].vertex)){
                if((vertice[vertice_index[1]].vertex-vertice[vertice_index[0]].vertex)*(vertice[vertice_index[1]].vertex-vertice[vertice_index[2]].vertex)<ZERO){
                    if_normal=1;
                    return;
                }
            }
        }
    }
    if_normal=0;
    return;
}

SimpleCell::SimpleCell()
{
    face_number=0;
    vertex_number=0;
    faces.clear();
    vertex_index.clear();
    if_normal=0;
    volume=0;
}

void SimpleCell::clear()
{
    face_number=0;
    vertex_number=0;
    faces.clear();
    vertex_index.clear();
    if_normal=0;
    volume=0;    
}

SimpleCell::SimpleCell(const std::vector<SimpleFace> &input_faces)
{
    faces=input_faces;
    face_number=faces.size();

	size_t temp=0;
    bool sign=0;
	for (auto i : faces) {
		for (auto j : i.vertice_index) {
			for (size_t k = 0; k < temp; k++) {
				if (vertex_index[k] == j) {
					sign = 1;
					break;
				}
			}
			if (sign == 1)
				sign = 0;
			else {
				vertex_index.push_back(j);
				temp++;
			}
		}
	}
    vertex_number=temp;

    if(face_number==6){
        for(size_t i=0;i<6;i++){
            if(faces[i].if_normal==0){
                if_normal=0;
                return;
            }
        }
        if_normal=1;
        return;
    }
    if_normal=0;
    return;
}

void SimpleCell::AddFace(const SimpleFace &input_face)
{
    faces.push_back(input_face);
    face_number=faces.size();
    bool sign=0;

    for(auto i:input_face.vertice_index){
		for (size_t k = 0; k < vertex_number; k++) {
			if (vertex_index[k] == i) {
				sign = 1;
            	break;
			}
		}
		if (sign == 1)
			sign = 0;
		else {
			vertex_index.push_back(i);
			vertex_number++;
		}        
    }
    if(face_number==6){
        for(size_t i=0;i<6;i++){
            if(faces[i].if_normal==0){
                if_normal=0;
                return;
            }
        }
        if_normal=1;
        return;
    }
    if_normal=0;
    return;
}

double SimpleCell::compute_cell_volume(const std::vector<SimpleVertex> &vertice)
{
    double my_volume=0;
    bool my_flag=0;
    for(auto i:faces){
        my_flag=0;
        for(auto j:i.vertice_index){
            if(j==vertex_index[0]){
                my_flag=1;
            }
        }
        if(my_flag==0){
            for(size_t j=1;j<i.vertex_number-1;j++){
                my_volume=my_volume+cxz::volume(vertice[vertex_index[0]].vertex,vertice[i.vertice_index[0]].vertex,vertice[i.vertice_index[j]].vertex,vertice[i.vertice_index[j+1]].vertex);
            }
        }
    }
    volume=my_volume;
    return volume;
}

size_t SimpleCell::vtk_format_number()
{
    size_t number=0;
    for(auto i:faces){
        number=number+i.vertex_number;
    }
    number=number+face_number+2;
    return number;
}

void SimpleCell::cell_write_to_file(ofstream &file)
{
    file<<vtk_format_number()-1<<" "<<face_number<<" ";
    for(auto i:faces){
        file<<i.vertex_number<<" ";
        for(auto j:i.vertice_index){
            file<<j<<" ";
        }
    }
    file<<std::endl;
}

SimpleVtk::SimpleVtk()
{
    vertice.clear();
    cells.clear();
    vertice_number=0;
    cells_number=0;
    volume=0;
    limit_volume=0;
}