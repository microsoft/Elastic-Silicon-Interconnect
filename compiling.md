# ESI compilation notes

## capnproto

```sh
git clone
cd c++
cmake -S. -Bbuild -DCMAKE_CXX_FLAGS:STRING=-fPIC
cmake --build build
sudo cmake --build build -t install
```
