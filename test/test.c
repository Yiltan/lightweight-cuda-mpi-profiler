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

  // Get the name of the processor
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  // Print off a hello world message
  printf("Hello world from processor %s, rank %d out of %d processors\n",
      processor_name, world_rank, world_size);

  int count = 64 * 1024 * 1024;
  float *sendbuf = (float*) malloc(count * sizeof(float));
  float *recvbuf = (float*) malloc(count * sizeof(float));

  // Init Data
  for (int i=0; i<count; i++) {
    sendbuf[i] = 1234.5678;
    recvbuf[i] = 0.0;
  }

  MPI_Allreduce(sendbuf, recvbuf, count, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  // Check data
  for (int i=0; i<count; i++) {
    float diff = abs(recvbuf[i] - (sendbuf[i] * world_size));

    if (diff > 0.01) {
      printf("Error In MPI_Allreduce()\n");
    }
  }

  // Finalize the MPI environment.
  MPI_Finalize();
}
