#include <iostream>
#include "Physika_Core/Arrays/array.h"
using namespace Physika;
using namespace std;
int main()
{
    int p[6] = {1,1,1,1,1,1};
    const Array<int> array(6, p);
    for (unsigned int i = 0; i < array.elementCount(); i++)
    {
        cout<<array[i];
        if(i != array.elementCount()-1)
            cout<<", ";
    }
    cout<<array<<endl;
    getchar();
    return 0;
}
