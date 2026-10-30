"""Single-process stand-in for mpi4py used only when OpenMPI/mpi4py is absent.

Official synthetic-iter/run.sh is `mpiexec -np 10 python main.py`.
MPI is used as independent replicas (rank for save filename / print gate),
not as a distributed SGD algorithm. This stub returns rank=0, size=1 so the
original SGD-GD loop in main.py can execute as a single replica.
"""


class _Comm:
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1


class MPI:
    COMM_WORLD = _Comm()
