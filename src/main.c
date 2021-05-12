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

#include "main.h"

int MPI_Init(int *argc, char ***argv) {
  printf("Inside %s\n", __func__);
  return PMPI_Init(argc, argv);
}

int MPI_Init_thread(int *argc, char ***argv, int required, int *provided) {
  printf("Inside %s\n", __func__);
  return PMPI_Init_thread(argc, argv, required, provided);
}

int MPI_Allreduce(const void *sendbuf, void *recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm) {
  printf("Inside %s\n", __func__);
  return PMPI_Allreduce(sendbuf, recvbuf, count, datatype, op, comm);
}

int MPI_Finalize(void) {
  printf("Inside %s\n", __func__);
  return PMPI_Finalize();
}
