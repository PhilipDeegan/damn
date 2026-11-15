#ifndef DAMN_HDF5_H5_FILE_HPP
#define DAMN_HDF5_H5_FILE_HPP

// HighFiveFile: an HDF5 file through HighFive, from PHARE's hdf5/detail/h5/h5_file.hpp
//
// opens serially or in parallel (MPI-IO, MPI_COMM_WORLD), reads and writes datasets
// (flat, nested vectors, chunked, resizable) and attributes. HDF5 needs every rank to
// create every dataset and attribute: the *_per_mpi functions gather paths, sizes and
// values with damn::mpi so each rank creates all of them, then writes only its own

#include "damn/core/def.hpp"
#include "damn/core/meta/container_meta.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/damn_mpi.hpp"

#include "highfive/H5File.hpp"
#include "highfive/H5Easy.hpp"

#include <string>
#include <string_view>
#include <cstdlib>
#include <vector>
#include <cstddef>
#include <algorithm>
#include <stdexcept>

namespace damn::hdf5::h5
{
using HiFile = HighFive::File;
using FileOp = HighFive::File::AccessMode;


template<typename Data, std::size_t dim>
NO_DISCARD auto vector_for_dim()
{
    if constexpr (dim == 1)
        return std::vector<Data>();
    if constexpr (dim == 2)
        return std::vector<std::vector<Data>>();
    if constexpr (dim == 3)
        return std::vector<std::vector<std::vector<Data>>>();
}

// the parallel driver: DAMN_HDF5_DRIVER=subfiling for HDF5's subfiling VFD (an I/O concentrator
// per node writing subfiles, needs MPI_THREAD_MULTIPLE), else MPI-IO. Asked for and not possible
// (HDF5 without it, MPI below MPI_THREAD_MULTIPLE): opening throws, saying why. Subfiled files
// are read through the same driver, or merged first (HDF5's h5fuse)
NO_DISCARD inline bool subfiling_requested()
{
    static bool const requested = [] {
        auto const* v = std::getenv("DAMN_HDF5_DRIVER");
        return v and std::string_view{v} == "subfiling";
    }();
    return requested;
}

#if defined(H5_HAVE_SUBFILING_VFD) && H5_HAVE_SUBFILING_VFD
// HighFive's FileAccessProps::add takes anything with apply(fapl)
struct SubfilingFileAccess
{
    MPI_Comm comm = MPI_COMM_WORLD;
    void apply(hid_t const fapl) const
    {
        if (H5Pset_mpi_params(fapl, comm, MPI_INFO_NULL) < 0
            or H5Pset_fapl_subfiling(fapl, nullptr) < 0) // defaults: a concentrator per node
            throw std::runtime_error("damn::hdf5: the subfiling file access properties failed");
    }
};
#endif

class HighFiveFile
{
public:
    template<typename FileAccessProps>
    static auto createHighFiveFile(std::string const path, FileOp flags, bool para,
                                   FileAccessProps& fapl)
    {
        if (para and subfiling_requested())
        {
#if defined(H5_HAVE_SUBFILING_VFD) && H5_HAVE_SUBFILING_VFD
            int level = 0;
            MPI_Query_thread(&level);
            if (level < MPI_THREAD_MULTIPLE)
                throw std::runtime_error("damn::hdf5: DAMN_HDF5_DRIVER=subfiling needs MPI at "
                                         "MPI_THREAD_MULTIPLE ("
                                         + path + ")");
            fapl.add(SubfilingFileAccess{});
            return HiFile{path, flags, fapl};
#else
            throw std::runtime_error("damn::hdf5: DAMN_HDF5_DRIVER=subfiling, HDF5 built without "
                                     "the subfiling VFD ("
                                     + path + ")");
#endif
        }
        if (para)
        {
#if defined(H5_HAVE_PARALLEL)
            fapl.add(HighFive::MPIOFileAccess{MPI_COMM_WORLD, MPI_INFO_NULL});
#else
            std::cout << "WARNING: PARALLEL HDF5 not available" << std::endl;
            if (mpi::size() > 1)
                throw std::runtime_error("HDF5 NOT PARALLEL!");
#endif
        }
        return HiFile{path, flags, fapl};
    }

    HighFiveFile(std::string const path, FileOp flags = HiFile::ReadWrite, bool para = true)
        : fapl_{}
        , h5file_{createHighFiveFile(path, flags, para, fapl_)}
    {
    }

    ~HighFiveFile() {}

    NO_DISCARD HiFile& file() { return h5file_; }


    template<typename T, std::size_t dim = 1>
    NO_DISCARD auto read_data_set_flat(std::string path) const
    {
        std::vector<T> data(H5Easy::getSize(h5file_, path));
        h5file_.getDataSet(path).read_raw(data.data());
        return data;
    }

    template<typename T, std::size_t dim = 1>
    NO_DISCARD auto read_data_set(std::string path) const
    {
        auto data = vector_for_dim<T, dim>();
        h5file_.getDataSet(path).read(data);
        return data;
    }


    template<std::size_t dim = 1, typename Data>
    auto& write_data_set(std::string path, Data const& data)
    {
        h5file_.getDataSet(path).write(data);
        return *this;
    }

    template<std::size_t dim = 1, typename Data>
    auto& write_data_set_flat(std::string path, Data const& data)
    {
        h5file_.getDataSet(path).write_raw(data);
        return *this;
    }


    template<typename Type, typename Size>
    auto create_data_set(std::string const& path, Size const& dataSetSize)
    {
        if (exist(path))
            return h5file_.getDataSet(path);
        createGroupsToDataSet(path);
        return h5file_.createDataSet<Type>(path, HighFive::DataSpace(dataSetSize));
    }


    template<typename Type>
    auto create_chunked_data_set(auto const& path, auto const chunk, auto const& dataspace)
    {
        if (exist(path))
            return h5file_.getDataSet(path);
        createGroupsToDataSet(path);
        HighFive::DataSetCreateProps props;
        props.add(HighFive::Chunking{chunk});
        H5Pset_fill_time(props.getId(), H5D_FILL_TIME_NEVER);
        return h5file_.createDataSet(path, dataspace, HighFive::create_datatype<Type>(), props);
    }

    template<typename Type, std::size_t cols>
    auto create_resizable_2d_data_set(auto const& path, hsize_t const x_chunk,
                                      hsize_t const y_chunk)
    {
        return create_chunked_data_set<Type>(
            path, std::vector<hsize_t>{x_chunk, y_chunk},
            HighFive::DataSpace({0, cols}, {HighFive::DataSpace::UNLIMITED, cols}));
    }

    template<typename Type>
    auto create_resizable_1d_data_set(auto const& path, hsize_t const chunk)
    {
        return create_chunked_data_set<Type>(
            path, std::vector<hsize_t>{chunk},
            HighFive::DataSpace({0}, {HighFive::DataSpace::UNLIMITED}));
    }


    /*
     * Communicate all dataset paths and sizes to all MPI process to allow each to create all
     * datasets independently. This is a requirement of HDF5.
     * in the case of a disparate number of datasets per MPI process, dataSetSize may be 0
     * such that the current process creates datasets for all other processes with non-zero
     * sizes. Recommended to use similar sized paths, if possible.
     */
    template<typename Type, typename Size>
    void create_data_set_per_mpi(std::string const& path, Size const& dataSetSize)
    {
        auto const mpi_size = mpi::size();
        auto const sizes    = mpi::collect(dataSetSize, mpi_size);
        auto const paths    = mpi::collect(path, mpi_size);
        for (int i = 0; i < mpi_size; i++)
        { // in the case an mpi node lacks something to write
            if (is_zero(sizes[i]) || paths[i] == "")
                continue;

            create_data_set<Type>(paths[i], sizes[i]);
        }
    }


    /*
     * Write attribute on all mpi cores, considered global, always the same path/key and value
     */
    template<typename Data>
    void write_attribute(std::string const& keyPath, std::string const& key, Data const& data)
    {
        // assumes all keyPaths and values are identical, and no null patches
        DAMN_DEBUG_DO({
            auto const paths = mpi::collect(keyPath, mpi::size());
            if (!std::all_of(paths.begin(), paths.end(),
                             [&](auto const& path) { return path == paths[0]; }))
                throw std::runtime_error("Function does not support different paths per mpi core");
        })

        constexpr bool data_is_vector = core::IsStdVector<Data>;

        auto doAttribute = [&](auto node, auto const& _key, auto const& value) {
            if constexpr (data_is_vector)
                node.template createAttribute<typename Data::value_type>(
                        _key, HighFive::DataSpace::From(value))
                    .write(value);
            else
                node.template createAttribute<Data>(_key, HighFive::DataSpace::From(value))
                    .write(value);
        };

        if (h5file_.exist(keyPath)
            && h5file_.getObjectType(keyPath) == HighFive::ObjectType::Dataset)
        {
            if (!h5file_.getDataSet(keyPath).hasAttribute(key))
                doAttribute(h5file_.getDataSet(keyPath), key, data);
        }
        else // group
        {
            createGroupsToDataSet(keyPath + "/dataset");
            if (!h5file_.getGroup(keyPath).hasAttribute(key))
                doAttribute(h5file_.getGroup(keyPath), key, data);
        }
    }



    /*
     * Communicate all attribute paths and values to all MPI process to allow each to create all
     * attributes independently. This is a requirement of HDF5.
     * in the case of a disparate number of attributes per MPI process, path may be an empty string
     * such that the current process creates attributes for all other processes with non-zero
     * sizes. Recommended to use similar sized paths, if possible. key is always assumed to the be
     * the same
     */
    template<typename Data>
    void write_attributes_per_mpi(std::string path, std::string key, Data const& data)
    {
        constexpr bool data_is_vector = core::IsStdVector<Data>;

        auto doAttribute = [&](auto node, auto const& _key, auto const& value) {
            if constexpr (data_is_vector)
            {
                if (value.size())
                    node.template createAttribute<typename Data::value_type>(
                            _key, HighFive::DataSpace(value.size()))
                        .write(value.data());
            }
            else
                node.template createAttribute<Data>(_key, HighFive::DataSpace::From(value))
                    .write(value);
        };

        int const mpi_size = mpi::size();
        auto const values  = [&]() {
            if constexpr (data_is_vector)
                return mpi::collect_raw(data, mpi_size);
            else
                return mpi::collect(data, mpi_size);
        }();
        auto const paths = mpi::collect(path, mpi_size);

        for (int i = 0; i < mpi_size; ++i)
        {
            std::string const keyPath = paths[i] == "null" ? "" : paths[i];
            if (keyPath.empty())
                continue;

            if (h5file_.exist(keyPath)
                && h5file_.getObjectType(keyPath) == HighFive::ObjectType::Dataset)
            {
                if (!h5file_.getDataSet(keyPath).hasAttribute(key))
                    doAttribute(h5file_.getDataSet(keyPath), key, values[i]);
            }
            else // group
            {
                createGroupsToDataSet(keyPath + "/dataset");
                if (!h5file_.getGroup(keyPath).hasAttribute(key))
                    doAttribute(h5file_.getGroup(keyPath), key, values[i]);
            }
        }
    }

    template<typename Attr = std::string>
    auto read_attribute(std::string const& path, std::string const& key)
    {
        auto at = h5file_.getGroup(path).getAttribute(key);
        Attr attr;
        at.read(attr);
        return attr;
    }

    bool exist(std::string const& s) const { return h5file_.exist(s); }
    auto getDataSet(std::string const& s)
    {
        if (!exist(s))
            throw std::runtime_error("Dataset does not exist: " + s);
        return h5file_.getDataSet(s);
    }

    HighFiveFile(HighFiveFile const&)            = delete;
    HighFiveFile(HighFiveFile&&)                 = delete;
    HighFiveFile& operator=(HighFiveFile const&) = delete;
    HighFiveFile& operator=(HighFiveFile&&)      = delete;

private:
    HighFive::FileAccessProps fapl_;
    HiFile h5file_;


    // during attribute/dataset creation, we currently don't require the parents of the group to
    // exist, but create all that are missing - this used to exist in highfive
    inline std::string getParentName(std::string const& path)
    {
        std::size_t idx = path.find_last_of("/");
        if (idx == std::string::npos or idx == 0)
            return "/";
        return path.substr(0, idx);
    }

    inline void createGroupsToDataSet(std::string const& path)
    {
        std::string group_name = getParentName(path);
        if (!h5file_.exist(group_name))
            h5file_.createGroup(group_name);
    }



    template<typename Size>
    static bool is_zero(Size size)
    {
        if constexpr (requires {
                          size.begin();
                          size.end();
                      })
            return std::all_of(size.begin(), size.end(), [](auto const& val) { return val == 0; });
        else
            return size == 0;
    }
};

} // namespace damn::hdf5::h5


#endif /* DAMN_HDF5_H5_FILE_HPP */
