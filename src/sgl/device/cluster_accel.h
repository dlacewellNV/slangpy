// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include "sgl/device/types.h"
#include "sgl/device/resource.h"

#include "sgl/core/enum.h"

#include <slang-rhi.h>

namespace sgl {

enum class ClusterAccelBuildOp : uint32_t {
    clas_from_triangles = static_cast<uint32_t>(rhi::ClusterAccelBuildOp::CLASFromTriangles),
    blas_from_clas = static_cast<uint32_t>(rhi::ClusterAccelBuildOp::BLASFromCLAS),
    templates_from_triangles = static_cast<uint32_t>(rhi::ClusterAccelBuildOp::TemplatesFromTriangles),
    clas_from_templates = static_cast<uint32_t>(rhi::ClusterAccelBuildOp::CLASFromTemplates),
};

SGL_ENUM_INFO(
    ClusterAccelBuildOp,
    {
        {ClusterAccelBuildOp::clas_from_triangles, "clas_from_triangles"},
        {ClusterAccelBuildOp::blas_from_clas, "blas_from_clas"},
        {ClusterAccelBuildOp::templates_from_triangles, "templates_from_triangles"},
        {ClusterAccelBuildOp::clas_from_templates, "clas_from_templates"},
    }
);
SGL_ENUM_REGISTER(ClusterAccelBuildOp);

struct ClusterAccelSizes {
    DeviceSize result_size{0};
    DeviceSize scratch_size{0};
};

struct ClusterAccelLimitsTriangles {
    uint32_t max_arg_count{0};
    uint32_t max_triangle_count_per_arg{0};
    uint32_t max_vertex_count_per_arg{0};
    uint32_t max_unique_sbt_index_count_per_arg{0};
    uint32_t position_truncate_bit_count{0};
};

struct ClusterAccelLimitsClusters {
    uint32_t max_arg_count{0};
    uint32_t max_total_cluster_count{0};
    uint32_t max_cluster_count_per_arg{0};
};

struct ClusterAccelBuildDesc {
    /// Operation to perform.
    ClusterAccelBuildOp op{ClusterAccelBuildOp::clas_from_triangles};

    /// Device buffer containing an array of op-specific device-args records written by kernels.
    BufferOffsetPair args_buffer{};
    /// Stride in bytes between consecutive arg records in args_buffer.
    uint32_t args_stride{0};
    /// Number of arg records to consume from args_buffer.
    uint32_t arg_count{0};

    /// Required in MVP: per‑op limits/hints to assist backends.
    /// Provide both structures for CLASFromTriangles and BLASFromCLAS.
    /// A value of 0 is invalid for required fields in MVP.
    ClusterAccelLimitsTriangles triangles_limits{};
    ClusterAccelLimitsClusters clusters_limits{};

    // Build mode and mode-specific parameters (defaults to Implicit)
    enum class BuildMode : uint32_t { implicit = 0, explicit_destinations = 1, get_sizes = 2 };
    BuildMode mode{BuildMode::implicit};

    struct ImplicitDesc {
        // Required output and temporary buffers for Implicit mode
        uint64_t output_buffer{0};
        uint64_t output_buffer_size_in_bytes{0};
        uint64_t temp_buffer{0};
        uint64_t temp_buffer_size_in_bytes{0};
        uint64_t output_handles_buffer{0};
        uint32_t output_handles_stride_in_bytes{0}; // 0 -> 8
        uint64_t output_sizes_buffer{0};
        uint32_t output_sizes_stride_in_bytes{0};   // 0 -> 4
    } implicit{};

    struct ExplicitDesc {
        // Required temporary buffer for Explicit mode
        uint64_t temp_buffer{0};
        uint64_t temp_buffer_size_in_bytes{0};
        uint64_t dest_addresses_buffer{0};         // required in Explicit
        uint32_t dest_addresses_stride_in_bytes{0}; // 0 -> 8
        uint64_t output_handles_buffer{0};          // 0 -> alias dest addresses
        uint32_t output_handles_stride_in_bytes{0}; // 0 -> 8
        uint64_t output_sizes_buffer{0};
        uint32_t output_sizes_stride_in_bytes{0};   // 0 -> 4
    } explicit_dest{};

    struct GetSizesDesc {
        // Required temporary buffer for GetSizes mode
        uint64_t temp_buffer{0};
        uint64_t temp_buffer_size_in_bytes{0};
        uint64_t output_sizes_buffer{0};
        uint32_t output_sizes_stride_in_bytes{0};   // 0 -> 4
    } get_sizes{};
};

} // namespace sgl


