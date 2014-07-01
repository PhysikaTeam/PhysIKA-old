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
 configfile.addOption("bool_d", &bool_d);
 configfile.addOption("float_e", &float_e);
 configfile.addOption("double_f", &double_f);
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