#pragma once

class SimpleVtk;

/**
 * @brief Separate the vtk object
 * 
 * @param origin_vtk 
 * @param new_vtk 
 * @return int 
 */
int separate_vtk(const SimpleVtk& origin_vtk, SimpleVtk& new_vtk);
