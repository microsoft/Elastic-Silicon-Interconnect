// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include <conio.h>
#include <dpi.hpp>
#include <iostream>
#include <thread>

using namespace std;

int main() {
  cout << "calling init()" << endl;
  sv2c_cosimserver_init();

  sv2c_cosimserver_ep_register(2, 6, 1024);
  cout << "waiting for newline" << endl;
  getchar();
  cout << "exiting" << endl;
  return 0;
}