from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

data = rank
if rank == 0:
    comm.send(data, dest=1, tag=11)
    data = comm.recv(source=1, tag=12)
elif rank == 1:
    data = comm.recv(source=0, tag=11)
    comm.send(data, dest=0, tag=12)

print(f'data={data} in process {rank}')
