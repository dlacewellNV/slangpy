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
    /// Required for clas_from_triangles and templates_from_triangles operations; must be non-zero.
    uint32_t max_arg_count{0};
    /// Required; maximum number of triangles in a single cluster.
    uint32_t max_triangle_count_per_arg{0};
    /// Required; maximum number of vertices in a single cluster.
    uint32_t max_vertex_count_per_arg{0};
    /// Required; maximum number of unique SBT indices within a single cluster.
    uint32_t max_unique_sbt_index_count_per_arg{0};
    /// Optional; minimum number of mantissa bits to truncate from vertex positions (0 means no truncation).
    uint32_t position_truncate_bit_count{0};
};

struct ClusterAccelLimitsClusters {
    /// Required for blas_from_clas operation; must be non-zero.
    uint32_t max_arg_count{0};
    /// Required; total number of cluster handles across all args.
    uint32_t max_total_cluster_count{0};
    /// Required; maximum number of cluster handles per arg.
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

    /// Per-operation limits. The active member is determined by the 'op' field.
    /// - clas_from_triangles: use limits_triangles
    /// - blas_from_clas: use limits_clusters
    /// - templates_from_triangles: use limits_triangles
    /// - clas_from_templates: use limits_triangles (reuses triangle limits, matching all backends)
    union {
        ClusterAccelLimitsTriangles limits_triangles;
        ClusterAccelLimitsClusters limits_clusters;
    } limits{};

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


