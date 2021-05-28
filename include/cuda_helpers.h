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

#ifndef CUDA_HELPERS_H
#define CUDA_HELPERS_H

#include "cuda_runtime.h"
#include <stdlib.h>

static inline int is_device_pointer(const void* ptr) {
  struct cudaPointerAttributes attributes;
  cudaError_t err = cudaPointerGetAttributes(&attributes, ptr);

  // Probabbly host pointer which has not been registered with CUDA API
  if (0 != err) {
    // This variable is unused. We collect this error so that it does not
    // interfere with the Application's error handling.
    #pragma GCC diagnostic ignored "-Wunused-variable"
    cudaError_t error = cudaGetLastError();
    return 0;
  }

  return (attributes.type == cudaMemoryTypeDevice) ||
         (attributes.type == cudaMemoryTypeManaged);
}

#endif // CUDA_HELPERS_H
