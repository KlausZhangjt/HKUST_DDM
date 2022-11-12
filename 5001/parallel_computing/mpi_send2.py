from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data={}

if rank == 0:
    data = {'a': 7, 'b': 3.14}
    req = comm.isend(data, dest=1, tag=11)
    req.wait()
elif rank == 1:
    req = comm.irecv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
    data = req.wait()

print(f'data={data} on process {rank}')
