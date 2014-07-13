/*
 * @file config_file_test.cpp 
 * @brief Test ConfigFile of Physika.
 * @author ShengYang
 * 
 * This file is part of Physika, a versatile physics simulation library.
 * Copyright (C) 2013 Physika Group.
 *
 * This Source Code Form is subject to the terms of the GNU General Public License v2.0. 
 * If a copy of the GPL was not distributed with this file, you can obtain one at:
 * http://www.gnu.org/licenses/gpl-2.0.html
 *
 */

#include <iostream>
#include "Physika_Core/Config_File/config_file.h"
#include <fstream>
using namespace Physika;
using namespace std;

int main()
{
    ConfigFile configfile;
    int a = 0, b = 0;
    string str_c = "";
    bool bool_d = false;
    float float_e = 0.0f;
    double double_f = 0.0;
    //cout<<a<<" "<<b<<" "<<str_c<<endl;

    configfile.addOptionOptional("a", &a, 3);
    configfile.addOption("b", &b);
    configfile.addOption("str_c", &str_c);
    // configfile.addOption("bool_d", &bool_d);
    configfile.addOption("float_e", &float_e);
    configfile.addOption("double_f", &double_f);
    configfile.addOptionOptional("bool_d", &bool_d, false);
    string file_name = "config_test.txt";

    if(configfile.parseFile(file_name) == 0)
    {
        cout<<"Parse File success!"<<endl;
    }
    else
    {
        cout<<"Parse File failed! Please check error informations!"<<endl;
    }


    configfile.printOptions();
    // cout<<"Hello test!"<<endl;
    cin>>a;
    return 0;
}
