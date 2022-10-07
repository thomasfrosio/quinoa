#include <iostream>

int main(int, char**) {
    int count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&count))
        return -1;
    if (count == 0)
        return -1;
    for (int device = 0; device < count; ++device) {
        if (device)
            std::cout << ';';
        cudaDeviceProp prop{};
        if (cudaSuccess == cudaGetDeviceProperties(&prop, device))
            std::cout << prop.major << prop.minor;
        else
            return -1;
    }
    return 0;
}
