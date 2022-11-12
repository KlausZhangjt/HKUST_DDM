from mpi4py import MPI
import numpy

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data=numpy.zeros(1)
data[0] = rank
data2=numpy.zeros(1)
data2[0] = rank
print(f'before reduction, data2 = {data2} on process {rank}', flush=True)
comm.Barrier()

comm.Reduce(data,data2,MPI.SUM, root=size-1)

comm.Barrier()

print(f'after reduction, data2 = {data2} on process {rank}', flush=True)
