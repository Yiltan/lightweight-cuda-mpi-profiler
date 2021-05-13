# lightweight-cuda-mpi-profiler

A Lightweight CUDA-Aware MPI Profiler built using the PMPI interface. 
This profiler is useful if you are trying to observe a specific property about an application which can be measure at the MPI API layer.
An example is givin in this repository but it can easily modified for your needs.

# Building The Project
It is required that you have an MPI implementation and CUDA installed. This was tested with OMPI 4.0.1 and CUDA 10.1.

Clone the repository:

```git clone https://github.com/Yiltan/lightweight-cuda-mpi-profiler.git```

Build the profiler:

``` cd lightweight-cuda-mpi-profiler && make ```

Check that the profiler was built correctly

```make check```

Now link this library to your application:

```mpicc -L<path_to_this_repositroy>/build/lib -llwcmp <program_file>.c -o application```

Now run your application with this profiler

```LD_LIBRARY_PATH=<path_to_this_repositroy>/build/lib:$(LD_LIBRARY_PATH) mpirun -np 4 ./application```

# How does this profiler work
## Summary of `include/main.h` 

In the file `include/main.h` we have a list of the MPI API calls that we wish to profile. In this example we have `MPI_Allreduce(...)`.
First we create a varible to track the metric we want to measure.

```int GPU_buffers_larger_than_4MB;```

Then we have a function to intialize the metric. Usually we set the varibles to 0 in this function.

```static inline void init_metrics()```

Next we have a function to increment our metrics as the application runs

```static inline void count_metrics()```

Finally we have a function to print our results

```static inline void print_metrics()``` 

## Summary of `src/main.c` 
Here is where we insert the code between the MPI API and the PMPI interface.

In the function `MPI_Init()` or `MPI_Init_thread()` we call our `init_metrics()` function to inizialize our variables.

As `MPI_Allreduce()` the API call which we plan to profile we add our `count_metrics()` function here.

To print our profiling results we add `print_metrics()` to `MPI_Finalize()` so that it is the last thing that occurs before the application finishes

# How can you modify it for your needs?
You can modify the metrics you wish to profile in `include/main.h`. If there are MPI API calls that you wish to profile, you can add them to `src/main.c`.

