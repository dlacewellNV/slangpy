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

struct ClusterAccelBuildDesc {
    /// Operation to perform.
    ClusterAccelBuildOp op{ClusterAccelBuildOp::clas_from_triangles};

    /// Device buffer containing an array of op-specific device-args records written by kernels.
    BufferOffsetPair args_buffer{};
    /// Stride in bytes between consecutive arg records in args_buffer.
    uint32_t args_stride{0};
    /// Number of arg records to consume from args_buffer.
    uint32_t arg_count{0};
};

} // namespace sgl


