#include "utility.h"

#ifdef WIN32
void check_dir(std::string& path)
{

    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        std::cout << "cannot access " << path << std::endl;
        std::cout << "mkdir: " << path << std::endl;
        std::cout << "mkdir: " << path << "screenshots" << std::endl;
        std::cout << "mkdir: " << path << "data" << std::endl;
        _mkdir(path.c_str());
        _mkdir((path + "screenshots").c_str());
        _mkdir((path + "data").c_str());
    }
    else if (info.st_mode & S_IFDIR)
        std::cout << path << " is a directory" << std::endl;
}

#elif __APPLE__
void check_dir(std::string& path)
{

    struct stat info;
    if (stat(path.c_str(), &info) != 0)
    {
        std::cout << "cannot access " << path << std::endl;
        std::cout << "mkdir: " << path << std::endl;
        std::cout << "mkdir: " << path << "screenshots" << std::endl;
        std::cout << "mkdir: " << path << "data" << std::endl;
        mkdir(path.c_str(), 0700);
        mkdir((path + "screenshots").c_str(), 0700);
        mkdir((path + "data").c_str(), 0700);
    }
    else if (info.st_mode & S_IFDIR)
        std::cout << path << " is a directory" << std::endl;
}

#endif