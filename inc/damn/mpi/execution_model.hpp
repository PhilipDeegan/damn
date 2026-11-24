#ifndef DAMN_MPI_EXECUTION_MODEL_HPP
#define DAMN_MPI_EXECUTION_MODEL_HPP

// damn::core::ExecutionModel's mpi_ranks from MPI: the size of MPI_COMM_WORLD if MPI is
// initialised, else 0

#include "damn/core/exec/execution_model.hpp"
#include "damn/mpi/mpi_utils.hpp"

#include <string>
#include <stdexcept>

namespace damn::mpi
{
// model's mpi_ranks from MPI
void set_ranks(core::ExecutionModel& model);

// model.check(), and its mpi_ranks those MPI has
void check(core::ExecutionModel const& model);

} // namespace damn::mpi


namespace damn::mpi
{
inline void set_ranks(core::ExecutionModel& model)
{
    model.mpi_ranks = is_init() ? size() : 0;
}

inline void check(core::ExecutionModel const& model)
{
    model.check();
    if (auto const ranks = is_init() ? size() : 0; model.mpi_ranks != ranks)
        throw std::runtime_error("damn::mpi::check: mpi_ranks " + std::to_string(model.mpi_ranks)
                                 + ", MPI has " + std::to_string(ranks));
}

} // namespace damn::mpi

#endif // DAMN_MPI_EXECUTION_MODEL_HPP
