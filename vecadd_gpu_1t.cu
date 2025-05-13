#include <iostream>
#include <math.h>
#include <chrono>
 
// Kernel function to add the elements of two arrays
__global__
void add(int n, float *sum, float *x, float *y)
{
  for (int i = 0; i < n; i++)
    sum[i] = x[i] + y[i];
}
 
int main(void)
{
 int N = 1<<29; // 512M elements
 
 // Allocate Unified Memory -- accessible from CPU or GPU
float *x, *y, *sum;
cudaMallocManaged(&x, N*sizeof(float));
cudaMallocManaged(&y, N*sizeof(float));

 
// Run kernel on 1M elements on the GPU
add<<<1, 1>>>(N, x, y);
 
// Wait for GPU to finish before accessing on host
cudaDeviceSynchronize();
 
 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;
 
// Free memory
cudaFree(x);
cudaFree(y);
 
 return 0;
}
