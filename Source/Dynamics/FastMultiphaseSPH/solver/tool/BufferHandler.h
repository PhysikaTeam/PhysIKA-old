#pragma once

class HostDeviceHandler
{
public:
    void* h_data;
    void* d_data;
    char  element_sz;  //char num per element
    int   sz;          //element num
};

class SortHandler
{
public:
    void* data;
    void* sorted_data;
    char  element_sz;  //char num per element
    int   sz;          //element num
};
