#ifndef _HEAD_
#define _HEAD_

//general use 
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <map>
#include <hash_map>
#include <unordered_map>
#include <queue>
#include <string>
#include <sstream>
#include <random>
#include <time.h>
#include <typeinfo>
// in C++ for C
#include <cstdio>
//for cuda 
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

//for assert() && static_assert()
#include <cassert> //for assert()
#include <type_traits> // for static_assert()

//for boost library no
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/info_parser.hpp>

//for opengMP
#include "omp.h"

//#define pi 3.1415926535898
#define __min__(a,b) (((a) < (b)) ? (a) : (b))

#endif
