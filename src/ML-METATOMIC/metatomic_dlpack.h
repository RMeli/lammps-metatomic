/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS Development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_METATOMIC_DLPACK_H
#define LMP_METATOMIC_DLPACK_H

/// DLPack wrappers for LAMMPS data structures.
///
/// The wrappers create *views* over memory that stays owned (and eventually
/// freed) by LAMMPS through the Memory class. The following invariants must
/// hold for the views to stay valid:
///
/// 1. LAMMPS buffers must outlive every object consuming their views
///    (metatensor Labels, TensorBlock, metatomic System). This is the case
///    when the buffers are members of classes persisting across MD steps
///    while the consuming objects only live for the duration of one step.
/// 2. Views must be rebuilt whenever the underlying buffer may have moved
///    (e.g. after a call to Memory::grow). Building a view is cheap: a few
///    dozen bytes of heap allocation.
///
/// The wrappers never free the data: the deleter of a view releases the
/// DLPack metadata (shape, strides, DLManagedTensorVersioned) only. The
/// single owner of the data remains the LAMMPS allocation.

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <vector>

#include <metatensor/dlpack/dlpack.h>
#include <metatensor.hpp>
#include <metatomic/utils.hpp>

namespace LAMMPS_NS {
namespace mta {

namespace details {

/// DLPack metadata for one view.
struct ViewCtx {
    std::vector<int64_t> shape;
    std::vector<int64_t> strides;
};

/// Deleter for a view over LAMMPS-owned data. It releases the DLPack
/// metadata only: the data itself is freed elsewhere by LAMMPS.
inline void view_deleter(DLManagedTensorVersioned *self)
{
    if (self == nullptr) return;
    delete static_cast<ViewCtx*>(self->manager_ctx);
    self->manager_ctx = nullptr;
    delete self;
}

/// Build non-owning DLPack view over LAMMPS-owned data
template <typename T>
DLManagedTensorVersioned *make_view(T *data, const std::vector<uintptr_t> &shape)
{
    auto ctx = std::make_unique<ViewCtx>();
    ctx->shape.reserve(shape.size());
    for (auto d : shape) ctx->shape.push_back(static_cast<int64_t>(d));

    const auto ndim = ctx->shape.size();
    ctx->strides.assign(ndim, 1);
    for (size_t i = ndim - 1; i > 0; --i)
        ctx->strides[i - 1] = ctx->strides[i] * ctx->shape[i];

    auto managed = std::make_unique<DLManagedTensorVersioned>();
    managed->version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
    managed->flags = 0; // read-write
    managed->manager_ctx = static_cast<void*>(ctx.get());
    managed->deleter = view_deleter;

    auto &tensor = managed->dl_tensor;
    tensor.device = DLDevice{kDLCPU, 0};
    tensor.dtype = metatensor::details::dtype_of<T>(); // TODO RM: Do not depend on metatensor implementation details?
    tensor.ndim = static_cast<int32_t>(ndim);
    tensor.byte_offset = 0;
    tensor.data = const_cast<void *>(static_cast<const void *>(data));
    tensor.shape = (ndim == 0) ? nullptr : ctx->shape.data();
    tensor.strides = (ndim == 0) ? nullptr : ctx->strides.data();

    ctx.release();
    return managed.release();
}

} // namespace details

/// Create a metatomic::DLPackTensor view over LAMMPS-owned data. The
/// tensor is row-major with the given shape, on the CPU device.
// TODO RM: Expand to GPU devices for Kokkos arrays
template <typename T>
metatomic::DLPackTensor dlpack_view(T *data, std::initializer_list<int64_t> shape)
{
    auto shape_ = std::vector<uintptr_t>();
    shape_.reserve(shape.size());
    for (auto d : shape) shape_.push_back(static_cast<uintptr_t>(d));

    return metatomic::DLPackTensor(details::make_view(data, shape_));
}

/// metatensor array view over LAMMPS-owned data.
template <typename T>
class LmpDataArray final : public metatensor::DataArrayBase {
public:
    /// Create a view with the given shape over `data`, which must be a
    /// row-major LAMMPS-owned buffer of matching size.
    LmpDataArray(T *data, std::vector<uintptr_t> shape) : data_(data), shape_(std::move(shape)) {}

    ~LmpDataArray() override = default;

    LmpDataArray(const LmpDataArray &) = default;
    LmpDataArray &operator=(const LmpDataArray &) = default;
    LmpDataArray(LmpDataArray &&) = default;
    LmpDataArray &operator=(LmpDataArray &&) = default;

    mts_data_origin_t origin() const override
    {
        mts_data_origin_t origin = 0;
        mts_register_data_origin("LAMMPS_NS::mta::LmpDataArray", &origin);
        return origin;
    }

    // TODO: Expand to GPU devices for Kokkos arrays
    DLDevice device() const override { return DLDevice{kDLCPU, 0}; }

    // TODO RM: Do not depend on metatensor details?
    DLDataType dtype() const override { return metatensor::details::dtype_of<T>(); }

    /// Get a DLPack representation of this array
    DLManagedTensorVersioned *as_dlpack(DLDevice device, const int64_t *stream,
                                      DLPackVersion max_version) override
    {
        // TODO: Expand to GPU devices for Kokkos arrays
        if (device.device_type != kDLCPU) {
            throw metatensor::Error("LmpDataArray only supports the CPU device");
        }
        if (stream != nullptr) {
            throw metatensor::Error("`stream` must be null for LmpDataArray (CPU data)");
        }
        if (max_version.major != DLPACK_MAJOR_VERSION) {
            throw metatensor::Error(
                "invalid `max_version` in LmpDataArray::as_dlpack: "
                "we got v" + std::to_string(max_version.major) + "." +
                std::to_string(max_version.minor) + ", but we support v" +
                std::to_string(DLPACK_MAJOR_VERSION) + "." +
                std::to_string(DLPACK_MINOR_VERSION));
        }
        return details::make_view(data_, shape_);
    }

    /// Create a new array from a DLPack tensor.
    std::unique_ptr<DataArrayBase> from_dlpack(DLManagedTensorVersioned *dl_tensor) const override
    {
        // Delete dl_tensor when leaving this function
        // Deletion occurs also if an exception is thrown
        struct AutoDelete
        {
            DLManagedTensorVersioned *ptr;
            ~AutoDelete()
            {
                if (ptr != nullptr && ptr->deleter != nullptr) { ptr->deleter(ptr); }
            }
        } guard{dl_tensor};

        if (dl_tensor == nullptr) {
            throw metatensor::Error("dlpack tensor is null in LmpDataArray::from_dlpack");
        }

        if (dl_tensor->version.major != DLPACK_MAJOR_VERSION) {
            throw metatensor::Error(
                "DLPack version mismatch in LmpDataArray::from_dlpack: expected major " +
                std::to_string(DLPACK_MAJOR_VERSION) + ", got " +
                std::to_string(dl_tensor->version.major));
        }

        // TODO: Expand to GPU devices for Kokkos arrays
        const auto &dl = dl_tensor->dl_tensor;
        if (dl.device.device_type != kDLCPU) {
            throw metatensor::Error(
                "LmpDataArray::from_dlpack only supports the CPU device, got device type " +
                std::to_string(dl.device.device_type));
        }
        if (dl.dtype.lanes != 1) {
            throw metatensor::Error("LmpDataArray::from_dlpack only supports lanes == 1");
        }
        // TODO RM: Do not depend on metatensor details?
        if (!metatensor::details::dlpack_dtype_matches<T>(dl.dtype)) {
            throw metatensor::Error("dtype mismatch in LmpDataArray::from_dlpack");
        }

        auto count = size_t{1};
        auto shape = std::vector<uintptr_t>();
        shape.reserve(static_cast<size_t>(dl.ndim));
        for (int32_t i = 0; i < dl.ndim; i++) {
            shape.push_back(static_cast<uintptr_t>(dl.shape[i]));
            count *= static_cast<size_t>(dl.shape[i]);
        }

        auto first = static_cast<const char *>(dl.data) + dl.byte_offset;
        auto data = std::vector<T>(reinterpret_cast<const T *>(first),
                                    reinterpret_cast<const T *>(first) + count);

        return std::unique_ptr<DataArrayBase>(
            new metatensor::SimpleDataArray<T>(std::move(shape), std::move(data)));
    }

    /// Make a copy of this array. The copy is a metatensor-owned
    /// SimpleDataArray living on the given device.
    std::unique_ptr<DataArrayBase> copy(DLDevice device) const override
    {
        if (device.device_type != kDLCPU) {
            throw metatensor::Error("LmpDataArray only supports copying to the CPU device");
        }

        auto count = size_t{1};
        for (auto d : shape_) count *= d;
        return std::unique_ptr<DataArrayBase>(new metatensor::SimpleDataArray<T>(
            shape_, std::vector<T>(data_, data_ + count)));
    }

    /// Create a new array with the same data type and location as this
    /// one, with the requested shape, filled with the scalar value from
    /// `fill_value`. The new array is a metatensor-owned SimpleDataArray.
    std::unique_ptr<DataArrayBase> create(std::vector<uintptr_t> shape,
                                        metatensor::MtsArray fill_value) const override
    {
        auto cpu_device = DLDevice{kDLCPU, 0};
        auto max_version = DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
        auto fill = fill_value.as_dlpack_array<T>(cpu_device, nullptr, max_version);

        if (!fill.shape().empty()) {
            throw metatensor::Error("`fill_value` must be a single scalar in LmpDataArray::create");
        }

        return std::unique_ptr<DataArrayBase>(
            new metatensor::SimpleDataArray<T>(std::move(shape), fill.data()[0]));
    }

    const std::vector<uintptr_t> &shape() const & override { return shape_; }

    void reshape(const std::vector<uintptr_t> & /*shape*/) override
    {
        throw metatensor::Error("reshape is not supported for LmpDataArray views");
    }

    void swap_axes(uintptr_t /*axis_1*/, uintptr_t /*axis_2*/) override
    {
        throw metatensor::Error("swap_axes is not supported for LmpDataArray views");
    }

    void move_data(const DataArrayBase & /*input*/, std::vector<mts_data_movement_t> /*moves*/) override
    {
        throw metatensor::Error("move_data is not supported for LmpDataArray views");
    }

    metatensor::NDArray<T> view() const
    {
        return metatensor::NDArray<T>(data_, shape_);
    }

    metatensor::NDArray<T> view() { return metatensor::NDArray<T>(data_, shape_); }

    /// Extract a reference to a LmpDataArray out of an mts_array_t.
    /// This function fails if the mts_array_t is not a LmpDataArray.
    static LmpDataArray &from_mts_array(metatensor::MtsArray &array)
    {
        auto origin = array.origin();

        std::array<char, 64> buffer = {0};
        auto status = mts_get_data_origin(origin, buffer.data(), buffer.size());
        if (status != MTS_SUCCESS || std::string(buffer.data()) != "LAMMPS_NS::mta::LmpDataArray") {
            throw metatensor::Error("this array is not a LAMMPS_NS::mta::LmpDataArray");
        }

        auto *base = static_cast<DataArrayBase *>(array.as_mts_array_t().ptr);
        return dynamic_cast<LmpDataArray &>(*base);
    }

private:
    /// Pointer to the data: LAMMPS-owned, never freed by this class.
    T *data_;
    /// Shape of the array.
    std::vector<uintptr_t> shape_;
};

} // namespace mta
} // namespace LAMMPS_NS

#endif
