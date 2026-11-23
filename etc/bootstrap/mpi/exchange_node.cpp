// bootstrap probe (see etc/bootstrap/bootstrap.py): ranks=2 profile=bootstrap mpirun=--map-by,core,--bind-to,core
// the cost of an exchange between two ranks on one node and one memory domain (Distance::Node):
// latency_s, s_per_byte (exchange.hpp)
#include "exchange.hpp"

int main(int argc, char** argv)
{
    return exchange_probe(argc, argv, /*same_numa=*/true);
}
