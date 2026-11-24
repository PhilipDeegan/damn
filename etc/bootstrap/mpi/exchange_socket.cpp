// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=2 profile=bootstrap mpirun=--map-by,numa,--bind-to,numa needs=mpi_found
// the cost of an exchange between two ranks on one node, other memory domains
// (Distance::Socket): latency_s, s_per_byte (exchange.hpp); fails on one memory domain
#include "exchange.hpp"

int main(int argc, char** argv)
{
    return exchange_probe(argc, argv, /*same_numa=*/false);
}
