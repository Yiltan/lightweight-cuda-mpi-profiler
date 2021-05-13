/* lightweight-cuda-mpi-profiler is a simple CUDA-Aware MPI profiler
 * Copyright (C) 2021 Yiltan Hassan Temucin
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#include "mpi.h"
#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
  // Initialize the MPI environment
  MPI_Init(NULL, NULL);

  // Get the number of processes
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  // Get the rank of the process
  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  int count = 8 * 1024 * 1024;
  size_t data_sz = count * sizeof(float);
  float *sendbuf_h = (float*) malloc(data_sz);
  float *recvbuf_h = (float*) malloc(data_sz);

  float *sendbuf_d, *recvbuf_d;
  cudaMalloc((void*) &sendbuf_d, data_sz);
  cudaMalloc((void*) &recvbuf_d, data_sz);

  // Init Data
  for (int i=0; i<count; i++) {
    sendbuf_h[i] = 1234.5678;
    recvbuf_h[i] = 0.0;
  }

  cudaMemcpy(sendbuf_d, sendbuf_h, data_sz, cudaMemcpyHostToDevice);
  cudaMemcpy(recvbuf_d, recvbuf_h, data_sz, cudaMemcpyHostToDevice);

  for (int i=0; i<20; i++) {
    MPI_Allreduce(sendbuf_d, recvbuf_d, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  }

  // Check data
  cudaMemcpy(recvbuf_h, recvbuf_d, data_sz, cudaMemcpyDeviceToHost);

  for (int i=0; i<count; i++) {
    float diff = abs(recvbuf_h[i] - (sendbuf_h[i] * world_size));

    if (diff > 0.01) {
      printf("Error In MPI_Allreduce()\n");
    }
  }

  // Finalize the MPI environment.
  MPI_Finalize();
}
