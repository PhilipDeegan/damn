#ifndef DAMN_HDF5_H5_FILE_HPP
#define DAMN_HDF5_H5_FILE_HPP

// an HDF5 file through HighFive, serial or MPI-IO: doc/hdf5/h5_file.md
// HDF5 needs every rank to create every dataset and attribute: the *_per_mpi functions gather
// paths, sizes and values so each rank creates all of them, then writes only its own

#include "damn/core/def.hpp"
#include "mkn/kul/meta.hpp"
#include "damn/mpi/mpi_utils.hpp"
#include "damn/mpi/damn_mpi.hpp"

#include "highfive/H5File.hpp"
#include "highfive/H5Easy.hpp"

#include <string>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace damn::hdf5::h5
{
using HiFile = HighFive::File;
using FileOp = HighFive::File::AccessMode;


template<typename Data, std::size_t dim>
MKN_KUL_FN_SIG_RET auto vector_for_dim()
{
    if constexpr (dim == 1)
        return std::vector<Data>();
    if constexpr (dim == 2)
        return std::vector<std::vector<Data>>();
    if constexpr (dim == 3)
        return std::vector<std::vector<std::vector<Data>>>();
}

// HDF5's subfiling VFD, DAMN_H5_SUB_FILES=<n> subfiles per file (0 or unset: MPI-IO):
// doc/hdf5/h5_file.md. Its concentrators are threads calling MPI: MPI_THREAD_MULTIPLE needed
MKN_KUL_FN_SIG_RET inline int sub_files()
{
    static int const n = [] {
        auto const* v = std::getenv("DAMN_H5_SUB_FILES");
        return v and *v ? std::stoi(v) : 0;
    }();
    return n;
}

MKN_KUL_FN_SIG_RET inline bool mpi_thread_multiple_required()
{
    return sub_files() > 0;
}

#if defined(H5_HAVE_SUBFILING_VFD) && H5_HAVE_SUBFILING_VFD
#include "H5FDsubfiling.h" // not in hdf5.h
// HighFive's FileAccessProps::add takes anything with apply(fapl)
struct SubfilingFileAccess
{
    int subfiles  = sub_files();
    MPI_Comm comm = MPI_COMM_WORLD;
    void apply(hid_t const fapl) const
    {
        H5FD_subfiling_config_t config{};
        if (H5Pset_mpi_params(fapl, comm, MPI_INFO_NULL) < 0
            or H5Pget_fapl_subfiling(fapl, &config) < 0) // the defaults, environment applied
            throw std::runtime_error("damn::hdf5: the subfiling defaults failed");
        config.shared_cfg.stripe_count = subfiles;
        auto const set                 = H5Pset_fapl_subfiling(fapl, &config);
        H5Pclose(config.ioc_fapl_id);
        if (set < 0)
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
        if (para and sub_files() > 0)
        {
#if defined(H5_HAVE_SUBFILING_VFD) && H5_HAVE_SUBFILING_VFD
            int level = 0;
            MPI_Query_thread(&level);
            if (level < MPI_THREAD_MULTIPLE)
                throw std::runtime_error("damn::hdf5: DAMN_H5_SUB_FILES needs MPI at "
                                         "MPI_THREAD_MULTIPLE ("
                                         + path + ")");
            fapl.add(SubfilingFileAccess{});
            return HiFile{path, flags, fapl};
#else
            throw std::runtime_error("damn::hdf5: DAMN_H5_SUB_FILES, HDF5 built without "
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

    MKN_KUL_FN_SIG_RET HiFile& file() { return h5file_; }


    template<typename T, std::size_t dim = 1>
    MKN_KUL_FN_SIG_RET auto read_data_set_flat(std::string path) const
    {
        std::vector<T> data(H5Easy::getSize(h5file_, path));
        h5file_.getDataSet(path).read_raw(data.data());
        return data;
    }

    template<typename T, std::size_t dim = 1>
    MKN_KUL_FN_SIG_RET auto read_data_set(std::string path) const
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


    // every rank's dataset created on every rank, as HDF5 needs; dataSetSize 0: this rank has
    //  none but still creates the others'
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


    // collective: the same path, key and value on every rank
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

        constexpr bool data_is_vector = mkn::kul::IsStdVector<Data>;

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



    // every rank's attribute created on every rank, as HDF5 needs; path "": this rank has none
    //  but still creates the others'. key the same on every rank
    template<typename Data>
    void write_attributes_per_mpi(std::string path, std::string key, Data const& data)
    {
        constexpr bool data_is_vector = mkn::kul::IsStdVector<Data>;

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


    // missing parent groups are created on the way: HighFive does not
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
