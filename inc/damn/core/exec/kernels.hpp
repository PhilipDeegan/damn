#ifndef DAMN_CORE_EXEC_KERNELS_HPP
#define DAMN_CORE_EXEC_KERNELS_HPP

#include "damn/core/def/damn_config.hpp"
#include "damn/core/exec/kernels/launchers.hpp"
#include "damn/core/def/detail/mkn_gpu.hpp"
#include "damn/core/exec/kernels.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace damn::core
{
template<typename Type, std::size_t dim>
struct Box;
} // namespace damn::core
namespace mkn::gpu
{
struct Guess;
} // namespace mkn::gpu

#if DAMN_HAVE_GPU || DAMN_HAVE_MKN_GPU

namespace damn::kernel
{

template<std::size_t dim, typename... Args>
void launch(core::Box<std::uint32_t, dim> const& box, std::size_t const& size, Args&&... args)
{
    // mkn::gpu launches when present, raja kernels are not implemented
    if constexpr (CompileOptions::WithMknGpu)
    {
        DAMN_WITH_MKN_GPU(
            { core::gpu::BoxCellNLauncher<core::Box<std::uint32_t, dim>>{box, size}(args...); });
    }
    else if constexpr (CompileOptions::WithRAJA)
    {
        throw std::runtime_error("launch::kernel NO IMPL");
    }
    else
        throw std::runtime_error("launch::kernel NO ALTERNATIVE");
}

template<typename... Args>
void launch(std::size_t const& size, Args&&... args)
{
    // mkn::gpu launches when present, raja kernels are not implemented
    if constexpr (CompileOptions::WithMknGpu)
    {
        DAMN_WITH_MKN_GPU({ core::gpu::DeviceLauncher<mkn::gpu::Guess, true>{size}(args...); });
    }
    else if constexpr (CompileOptions::WithRAJA)
    {
        throw std::runtime_error("launch::kernel NO IMPL");
    }
    else
        throw std::runtime_error("launch::kernel NO ALTERNATIVE");
}

// same as above, on a stream, waits for that stream only
template<typename Stream, typename... Args>
    requires(!std::is_convertible_v<Stream const&, std::size_t>)
void launch(Stream& stream, std::size_t const& size, Args&&... args)
{
    // mkn::gpu launches when present, raja kernels are not implemented
    if constexpr (CompileOptions::WithMknGpu)
    {
        DAMN_WITH_MKN_GPU({
            core::gpu::DeviceLauncher<mkn::gpu::Guess, true>{size}.stream(stream, args...);
        });
    }
    else if constexpr (CompileOptions::WithRAJA)
    {
        throw std::runtime_error("launch::kernel NO IMPL");
    }
    else
        throw std::runtime_error("launch::kernel NO ALTERNATIVE");
}


} // namespace damn::kernel

#else

namespace damn::kernel
{
template<typename... Args>
void launch(Args&&... args) // noop
{
    throw std::runtime_error("nooop");
}

} // namespace damn::kernel

#endif // DAMN_HAVE_GPU || DAMN_HAVE_MKN_GPU

#endif /* DAMN_CORE_EXEC_KERNELS_HPP */
