#pragma once

#include <sys/types.h>
#include <sys/stat.h>
#include <string>
#include <iostream>

#ifdef WIN32
#include <direct.h>
#elif __APPLE__
#include <unistd.h>
#define _mkdir mkdir
#endif

void check_dir(std::string& path);