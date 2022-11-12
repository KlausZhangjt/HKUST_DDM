from mpi4py import MPI

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = (rank+1)**2
print(f'before gathering, data = {data} on process {rank}', flush=True)
comm.Barrier()

data = comm.gather(data, root=0)

comm.Barrier()

print(f'after gathering, data = {data} on process {rank}', flush=True)
