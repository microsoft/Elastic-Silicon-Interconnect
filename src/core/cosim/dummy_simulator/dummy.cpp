#include <dpi.hpp>
#include <thread>
#include <iostream>

using namespace std;

int main()
{
    cout << "calling init()" << endl;
    sv2c_cosimserver_init();
    cout << "sleeping... " << endl;
    std::this_thread::sleep_for(std::chrono::seconds(10));
    cout << "exiting... " << endl;
    return 0;
}