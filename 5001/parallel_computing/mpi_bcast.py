from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = rank
print(f'before broadcasting, data = {data} on process {rank}', flush=True)
comm.Barrier()

data = comm.bcast(data, root=size-1)

comm.Barrier()

print(f'after broadcasting, data = {data} on process {rank}', flush=True)
