/*
 * @file obj_mesh_io.cpp 
 * @brief load and save mesh to an obj file.
 * @author Fei Zhu
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <sstream>
#include <fstream>
#include "Physika_Core/Utilities/physika_assert.h"
#include "Physika_Geometry/Surface_Mesh/surface_mesh.h"
#include "Physika_IO/Surface_Mesh_IO/obj_mesh_io.h"

namespace Physika{

template <typename Scalar>
void ObjMeshIO<Scalar>::load(const string &filename, SurfaceMesh<Scalar> *mesh)
{
	PHYSIKA_ASSERT(mesh);
	unsigned int num_face=0;

	const int maxline =1000;
	std::fstream ifs(filename.c_str());
	char line[maxline];

	Group<Scalar>* current_group=NULL;   ///careful
	unsigned int current_material_index=0;

	string::size_type suffix_idx = filename.find('.');
    PHYSIKA_ASSERT(suffix_idx < filename.size());
    string suffix = filename.substr(suffix_idx);

	if(suffix!=string(".obj")){cout<<"The file you load in is not a obj file. Please reload!"<<endl;PHYSIKA_ASSERT(0);}

	if(!ifs)
	{
		string message("couldn't open .obj file");
		PHYSIKA_ASSERT(0);

	}

	int line_num =0;
	int num_group_faces = 0;
	int group_clone_index = 0;
	std::string group_source_name;

	while(ifs){
			line_num++;
			ifs.getline(line,maxline);

	

			//follow  :   clear withspace to single blanks in every line

	
			char *p=line;
			while(*p){
				while((*p == ' ')&&(*(p+1) == 0 || *(p+1) == ' ')){
					char *q=p;
					while(*q){
						*q=*(q+1);
						q++;
					}
				}
				p++;
			}
		

		char commond =line [0];
		std::stringstream stream;
		stream.clear();
		char type_of_line[50];
		if(strncmp(line,"v ",2) == 0){
			//vertex
			Scalar x,y,z;
			stream.clear();
			stream<<line;
			stream>>type_of_line;
			PHYSIKA_ASSERT(stream>>x);
			PHYSIKA_ASSERT(stream>>y);
			PHYSIKA_ASSERT(stream>>z);
			mesh->addVertexPosition(Vector<Scalar,3>(x,y,z));
		}
		else if(strncmp(line, "vn ", 3) == 0){
			Scalar x,y,z;
			stream.clear();
			stream<<line;
			stream>>ttype_of_line;
			PHYSIKA_ASSERT(stream>>x);
			PHYSIKA_ASSERT(stream>>y);
			PHYSIKA_ASSERT(stream>>z);
			mesh->addVertexNormal(Vector<Scalar,3>(x,y,z));
		}
		else if(strncmp(line, "vt", 3)==0){
			Scalar x,y;
			stream.clear();
			stream<<line;
			stream>>ttype_of_line;
			PHYSIKA_ASSERT(stream>>x);
			PHYSIKA_ASSERT(stream>>y);
			mesh->addVertexTextureCoordinate(Vector<Scalar,2>(x,y));
		}
		else if(strncmp(line, "g ", 2) == 0){
			char s[4096]={};
			stream.clear();
			stream<<line;
			stream>>ttype_of_line;
			stream>>s;
			if(strlen(s)<1)cout<<"warning: empty group name come in"<<endl;
			PHYSIKA_ASSERT(strlen(s));
			string group_name(s);
			if(current_group=mesh->groupPtr(group_name)){
				
			}
			else{
				mesh->addGroup(Group<Scalar>(group_name,current_material_index));
				current_group=mesh->groupPtr(group_name);
				group_source_name=group_name;
				group_clone_index=0;
				num_group_faces=0;
			}
			
		}
		else if(strncmp(line, "f ",2) == 0|| (strncmp(line, "fo ", 3) == 0)){
			stream.clear();
			stream<<line;
			stream>>ttype_of_line;
			if(current_group==NULL){
				mesh->addGroup(Group<Scalar>(string("default")));
				current_group=mesh->groupPtr(string("default"));
			}
			
			Face<Scalar> face_temple;
			char vertex_indice[20]={};
			while(stream>>vertex_indice){
				unsigned int pos;
				unsigned int nor;
				unsigned int tex;
				
				if(strstr(vertex_indice,"//") != NULL){
					// v//n
					if(sscanf(vertex_indice,"%u//%u", &pos, &nor) == 2 ){
						Vertex<Scalar> vertex_temple(pos-1);
						vertex_temple.setNormalIndex(nor-1);
						face_temple.addVertex(vertex_temple);
					}
					else PHYSIKA_ASSERT("invalid vertx in this face\n");
						
				}
				else{
					if(sscanf(vertex_indice,"%u/%u/%u", &pos,&tex,&nor) != 3){
						if(strstr(vertex_indice,"/") != NULL){
							//  v/t
							
							if(sscanf(vertex_indice, "%u/%u", &pos, &tex) == 2){
			                    Vertex<Scalar> vertex_temple(pos-1);
								vertex_temple.setTextureCoordinateIndex(tex-1);
								face_temple.addVertex(vertex_temple);
							}
							else{
								PHYSIKA_ASSERT("%u/%u error");
							}
						}
						else {
							if(sscanf(vertex_indice,"%u", &pos) == 1){
								face_temple.addVertex(Vertex<Scalar>(pos-1));
							}
							else {
								PHYSIKA_ASSERT("%u error");
							}
						}
					}
					else {
						Vertex<Scalar> vertex_temple(pos-1);
						vertex_temple.setNormalIndex(nor-1);
						vertex_temple.setTextureCoordinateIndex(tex-1);
						face_temple.addVertex(vertex_temple);
					}
				}
			}// end while vertex_indices
			num_face++;
			num_group_faces++;
			current_group->addFace(face_temple);
		}
		else if((strncmp(line, "#", 1) == 0) || (strncmp(line, "\0", 1) == 0)){
			
		}
		else if(strncmp(line, "usemtl", 6) == 0){
			if (num_group_faces > 0)
			{
				// usemtl without a "g" statement : must create a new group
				//first ,create unique name
				char newNameC[4096]={};
				sprintf(newNameC, "%s.%d", group_source_name.c_str(),group_clone_index);
	
				std::string newName(newNameC);
				mesh->addGroup(Group<Scalar>(newName,current_material_index));
				current_group=mesh->groupPtr(newName);
				num_group_faces = 0;
				group_clone_index++;	
			}

			bool materialfound =false;
			unsigned int counter = 0;
			char material_name[128]={};
			stream.clear();
			stream<<line;
			stream>>type_of_line;
			stream>>material_name;
			if(mesh->materialPtr(string(material_name))){
				if(mesh->groups_.empty()){
					groups.push_back(Group<Scalar>(string("default") ) );
					current_group = mesh->groupPtr(string("default"));
				}
				current_group->setMaterialIndex(current_material_index);
			}
			else {PHYSIKA_ASSERT("material found false");}
		}
		else if(strncmp(line, "mtllib", 6) == 0){
			// implement
			stream.clear();
			stream<<line;
			stream>>ttype_of_line;
			char mtl_name[4096];
			stream>>mtl_name;
			loadMaterials(string(mtl_name),mesh);
		}
		else {
			// do nothing
		}		
	}//end while
	ifs.close();
	//cout  file message



}

template <typename Scalar>
void ObjMeshIO<Scalar>::save(const string &filename, SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
	std::fstream fileout(filename.c_str());


}

template <typename Scalar>
void ObjMeshIO<Scalar>::loadMaterials(const string &filename, SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
}

template <typename Scalar>
void ObjMeshIO<Scalar>::saveMaterials(const string &filename, SurfaceMesh<Scalar> *mesh)
{
//TO DO: implementation
}

} //end of namespace Physika



















