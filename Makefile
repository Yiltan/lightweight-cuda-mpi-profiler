#lightweight-cuda-mpi-profiler is a simple CUDA-Aware MPI profiler
#Copyright (C) 2021 Yiltan Hassan Temucin
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <https://www.gnu.org/licenses/>.

CC=mpicc

all: build/obj/.d build/lib/.d
	$(CC) -Iinclude/ -c src/main.c -o build/obj/main.o
	$(CC) -shared -o build/lib/liblwcmp.so build/obj/main.o

clean:
	rm -rf build/*

%/.d:
	mkdir -p $(@D)
	touch $@
