from mpi4py import MPI

# %% question 1
# Commands in terminal ro run this python file:
# cd /Users/klaus_zhangjt/PycharmProjects/pythonProject1
# chmod a+x MPI.py

# Commands in terminal to repeat the circle and record total time cost
# #!/bin/bash
# start=$(date +%s)
#
# for i in {1..50}; do mpirun -n 7 python3 MPI.py; done;
#
# end=$(date +%s)
# take=$(( end - start ))
# echo Total time cost is ${take} seconds.

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
data = 'Hey!'

if rank == 0:
    req = comm.isend(data, dest=1, tag=0)
    req.wait()
    print('Process {} sent to Process {}:'.format(rank, rank+1), data)
    req = comm.irecv(source=size - 1, tag=size - 1)
    req.wait()
    print('Process {} received from Process{}:'.format(rank, size-1), data)
for i in range(1, size):
    if rank == i and i < size-1:
        req = comm.irecv(source=i - 1, tag=i-1)
        req.wait()
        print('Process {} received from Process{}:'.format(rank, rank-1), data)
        req = comm.isend(data, dest=i + 1, tag=i)
        req.wait()
        print('Process {} sent to Process {}:'.format(rank, rank+1), data)
    if rank == i and i == size - 1:
        req = comm.irecv(source=size - 2, tag=size-2)
        req.wait()
        print('Process {} received from Process{}:'.format(rank, rank-1), data)
        req = comm.isend(data, dest=0, tag=size-1)
        req.wait()
        print('Process {} sent to Process {}:'.format(rank, 0), data)
