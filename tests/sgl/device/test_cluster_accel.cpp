// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "testing.h"

#include "sgl/device/device.h"
#include "sgl/device/cluster_accel.h"
// Minimal OptiX stand-ins to let the smoke test compile without OptiX headers.
using CUdeviceptr = unsigned long long;
enum { OPTIX_CLUSTER_ACCEL_CLUSTER_FLAG_NONE = 0 };
enum { OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_32BIT = 4 };
struct OptixClusterAccelPrimitiveInfo { unsigned int sbtIndex:24; unsigned int reserved:5; unsigned int primitiveFlags:3; };
struct OptixClusterAccelBuildInputTrianglesArgs
{
    unsigned int clusterId;
    unsigned int clusterFlags;
    unsigned int triangleCount              : 9;
    unsigned int vertexCount                : 9;
    unsigned int positionTruncateBitCount   : 6;
    unsigned int indexFormat                : 4;
    unsigned int opacityMicromapIndexFormat : 4;
    OptixClusterAccelPrimitiveInfo basePrimitiveInfo;
    unsigned short indexBufferStrideInBytes;
    unsigned short vertexBufferStrideInBytes;
    unsigned short primitiveInfoBufferStrideInBytes;
    unsigned short opacityMicromapIndexBufferStrideInBytes;
    CUdeviceptr indexBuffer;
    CUdeviceptr vertexBuffer;
    CUdeviceptr primitiveInfoBuffer;
    CUdeviceptr opacityMicromapArray;
    CUdeviceptr opacityMicromapIndexBuffer;
    CUdeviceptr instantiationBoundingBoxLimit;
};
struct OptixClusterAccelBuildInputClustersArgs
{
    unsigned int clusterHandlesCount;
    unsigned int clusterHandlesBufferStrideInBytes;
    CUdeviceptr  clusterHandlesBuffer;
};

using namespace sgl;
// Forward declarations (bring CommandEncoder & CommandBuffer into view for this TU)
#include "sgl/device/command.h"

TEST_SUITE_BEGIN("cluster_accel");

TEST_CASE("optix_sizes")
{
    // Create CUDA device specifically for OptiX tests.
    ref<Device> device;
    try {
        device = Device::create({.type = DeviceType::cuda, .enable_debug_layers = false});
    } catch (...) {
        // Skip silently if CUDA device creation fails.
        return;
    }

    // Require OptiX v9+ and feature support.
    if (!device || device->info().optix_version < 90000
        || !device->has_feature(Feature::cluster_acceleration_structure))
    {
        // Skip silently if feature/version unsupported.
        return;
    }

    // Triangles -> CLAS sizes
    ClusterAccelBuildDesc clas_desc = {};
    clas_desc.op = ClusterAccelBuildOp::clas_from_triangles;
    clas_desc.triangles_limits.max_arg_count = 1;
    clas_desc.triangles_limits.max_triangle_count_per_arg = 1;
    clas_desc.triangles_limits.max_vertex_count_per_arg = 3;
    clas_desc.triangles_limits.max_unique_sbt_index_count_per_arg = 1;
    ClusterAccelSizes clas_sizes = device->get_cluster_acceleration_structure_sizes(clas_desc);
    INFO("CLAS sizes result=" << clas_sizes.result_size << ", scratch=" << clas_sizes.scratch_size);
    CHECK(clas_sizes.result_size > 0);
    CHECK(clas_sizes.scratch_size > 0);

    // CLAS -> BLAS sizes
    ClusterAccelBuildDesc blas_desc = {};
    blas_desc.op = ClusterAccelBuildOp::blas_from_clas;
    blas_desc.clusters_limits.max_arg_count = 1;
    blas_desc.clusters_limits.max_total_cluster_count = 1;
    blas_desc.clusters_limits.max_cluster_count_per_arg = 1;
    ClusterAccelSizes blas_sizes = device->get_cluster_acceleration_structure_sizes(blas_desc);
    INFO("BLAS sizes result=" << blas_sizes.result_size << ", scratch=" << blas_sizes.scratch_size);
    CHECK(blas_sizes.result_size > 0);
    CHECK(blas_sizes.scratch_size > 0);
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("cluster_accel_build");

TEST_CASE("optix_build_one_triangle")
{
    // Create CUDA device
    ref<Device> device;
    try {
        device = Device::create({.type = DeviceType::cuda, .enable_debug_layers = false});
    } catch (...) {
        return;
    }
    if (!device || device->info().optix_version < 90000
        || !device->has_feature(Feature::cluster_acceleration_structure))
    {
        return;
    }

    struct Float3 { float x, y, z; };

    // Geometry buffers
    const Float3 vertices[3] = {{0.f, 0.f, 0.f}, {1.f, 0.f, 0.f}, {0.f, 1.f, 0.f}};
    const uint32_t indices[3] = {0, 1, 2};

    ref<Buffer> vbuf = device->create_buffer({
        .size = sizeof(vertices),
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure_build_input,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "tri-vertices",
        .data = vertices,
        .data_size = sizeof(vertices),
    });
    ref<Buffer> ibuf = device->create_buffer({
        .size = sizeof(indices),
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure_build_input,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "tri-indices",
        .data = indices,
        .data_size = sizeof(indices),
    });

    // One triangles args record
    // TODO: Replace with API-agnostic device ABI struct (cluster_accel_abi.slang) when available.
    OptixClusterAccelBuildInputTrianglesArgs tri_args = {};
    tri_args.clusterId = 0;
    tri_args.clusterFlags = OPTIX_CLUSTER_ACCEL_CLUSTER_FLAG_NONE;
    tri_args.triangleCount = 1;
    tri_args.vertexCount = 3;
    tri_args.positionTruncateBitCount = 0;
    tri_args.indexFormat = OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_32BIT; // 4 bytes per index
    tri_args.opacityMicromapIndexFormat = 0; // none
    tri_args.basePrimitiveInfo = {};
    tri_args.indexBufferStrideInBytes = 0; // natural
    tri_args.vertexBufferStrideInBytes = sizeof(Float3);
    tri_args.primitiveInfoBufferStrideInBytes = 0;
    tri_args.opacityMicromapIndexBufferStrideInBytes = 0;
    tri_args.indexBuffer = static_cast<CUdeviceptr>(ibuf->device_address());
    tri_args.vertexBuffer = static_cast<CUdeviceptr>(vbuf->device_address());
    tri_args.primitiveInfoBuffer = 0;
    tri_args.opacityMicromapArray = 0;
    tri_args.opacityMicromapIndexBuffer = 0;
    tri_args.instantiationBoundingBoxLimit = 0;

    ref<Buffer> args = device->create_buffer({
        .size = sizeof(OptixClusterAccelBuildInputTrianglesArgs),
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure_build_input,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "clas-args"
    });
    args->set_data(&tri_args, sizeof(tri_args), 0);

    // Sizes for CLASFromTriangles
    ClusterAccelBuildDesc clas_desc = {};
    clas_desc.op = ClusterAccelBuildOp::clas_from_triangles;
    clas_desc.args_buffer = BufferOffsetPair(args);
    clas_desc.args_stride = sizeof(OptixClusterAccelBuildInputTrianglesArgs);
    clas_desc.arg_count = 1;
    clas_desc.triangles_limits.max_arg_count = 1;
    clas_desc.triangles_limits.max_triangle_count_per_arg = 1;
    clas_desc.triangles_limits.max_vertex_count_per_arg = 3;
    clas_desc.triangles_limits.max_unique_sbt_index_count_per_arg = 1;
    ClusterAccelSizes clas_sizes = device->get_cluster_acceleration_structure_sizes(clas_desc);

    // Allocate result/scratch
    ref<Buffer> result = device->create_buffer({
        .size = clas_sizes.result_size,
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "clas-output",
    });
    ref<Buffer> scratch = device->create_buffer({
        .size = clas_sizes.scratch_size,
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "clas-scratch",
    });

    // Build
    ref<CommandEncoder> enc = device->create_command_encoder();
    enc->build_cluster_acceleration_structure(
        clas_desc,
        BufferOffsetPair(scratch),
        BufferOffsetPair(result)
    );
    ref<CommandBuffer> cb = enc->finish();
    device->submit_command_buffer(cb.get());
    device->wait();

    // Verify non-zero first handle/size entry (8 bytes)
    uint64_t first_qword = 0;
    device->read_buffer_data(result.get(), &first_qword, sizeof(first_qword), 0);
    CHECK(first_qword != 0);

    // Now build BLAS from the produced CLAS handle
    OptixClusterAccelBuildInputClustersArgs clusters_args = {};
    clusters_args.clusterHandlesCount = 1;
    clusters_args.clusterHandlesBufferStrideInBytes = 8;
    clusters_args.clusterHandlesBuffer = static_cast<CUdeviceptr>(result->device_address());

    ref<Buffer> blas_args_buf = device->create_buffer({
        .size = sizeof(OptixClusterAccelBuildInputClustersArgs),
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure_build_input,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "blas-args"
    });
    blas_args_buf->set_data(&clusters_args, sizeof(clusters_args), 0);

    ClusterAccelBuildDesc blas_desc = {};
    blas_desc.op = ClusterAccelBuildOp::blas_from_clas;
    blas_desc.args_buffer = BufferOffsetPair(blas_args_buf);
    blas_desc.args_stride = sizeof(OptixClusterAccelBuildInputClustersArgs);
    blas_desc.arg_count = 1;
    blas_desc.clusters_limits.max_arg_count = 1;
    blas_desc.clusters_limits.max_total_cluster_count = 1;
    blas_desc.clusters_limits.max_cluster_count_per_arg = 1;

    ClusterAccelSizes blas_sizes = device->get_cluster_acceleration_structure_sizes(blas_desc);
    ref<Buffer> blas_result = device->create_buffer({
        .size = blas_sizes.result_size,
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "blas-output",
    });
    ref<Buffer> blas_scratch = device->create_buffer({
        .size = blas_sizes.scratch_size,
        .memory_type = MemoryType::device_local,
        .usage = BufferUsage::acceleration_structure,
        .default_state = ResourceState::acceleration_structure_build_output,
        .label = "blas-scratch",
    });

    enc = device->create_command_encoder();
    enc->build_cluster_acceleration_structure(
        blas_desc,
        BufferOffsetPair(blas_scratch),
        BufferOffsetPair(blas_result)
    );
    cb = enc->finish();
    device->submit_command_buffer(cb.get());
    device->wait();

    uint64_t blas_first_qword = 0;
    device->read_buffer_data(blas_result.get(), &blas_first_qword, sizeof(blas_first_qword), 0);
    CHECK(blas_first_qword != 0);
}

// -----------------------------------------------------------------------------
// Batch CLAS/BLAS build smoke
// -----------------------------------------------------------------------------

TEST_CASE("optix_batch_build_two_clusters")
{
    ref<Device> device;
    try { device = Device::create({.type = DeviceType::cuda}); } catch (...) { return; }
    if (!device || device->info().optix_version < 90000 || !device->has_feature(Feature::cluster_acceleration_structure)) return;

    struct Float3 { float x, y, z; };
    const Float3 vertices[6] = {{0,0,0},{1,0,0},{0,1,0}, {2,0,0},{3,0,0},{2,1,0}};
    // Indices per cluster must be local to the provided vertex buffer base
    const uint32_t indices[6] = {0,1,2, 0,1,2};

    ref<Buffer> vbuf = device->create_buffer({.size=sizeof(vertices), .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure_build_input, .default_state=ResourceState::acceleration_structure_build_output, .label="tri-vertices", .data=vertices, .data_size=sizeof(vertices)});
    ref<Buffer> ibuf = device->create_buffer({.size=sizeof(indices), .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure_build_input, .default_state=ResourceState::acceleration_structure_build_output, .label="tri-indices", .data=indices, .data_size=sizeof(indices)});

    OptixClusterAccelBuildInputTrianglesArgs tri_args[2] = {};
    for (int i=0;i<2;i++) {
        tri_args[i].clusterId = (unsigned)i;
        tri_args[i].clusterFlags = OPTIX_CLUSTER_ACCEL_CLUSTER_FLAG_NONE;
        tri_args[i].triangleCount = 1;
        tri_args[i].vertexCount = 3;
        tri_args[i].positionTruncateBitCount = 0;
        tri_args[i].indexFormat = OPTIX_CLUSTER_ACCEL_INDICES_FORMAT_32BIT;
        tri_args[i].opacityMicromapIndexFormat = 0;
        tri_args[i].basePrimitiveInfo = {};
        tri_args[i].indexBufferStrideInBytes = 0;
        tri_args[i].vertexBufferStrideInBytes = sizeof(Float3);
        tri_args[i].primitiveInfoBufferStrideInBytes = 0;
        tri_args[i].opacityMicromapIndexBufferStrideInBytes = 0;
        tri_args[i].indexBuffer = (CUdeviceptr)ibuf->device_address() + (i*3*sizeof(uint32_t));
        tri_args[i].vertexBuffer = (CUdeviceptr)vbuf->device_address() + (i*3*sizeof(Float3));
    }

    ref<Buffer> args = device->create_buffer({.size=sizeof(tri_args), .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure_build_input, .default_state=ResourceState::acceleration_structure_build_output, .label="clas-args-batch"});
    args->set_data(&tri_args[0], sizeof(tri_args), 0);

    ClusterAccelBuildDesc clas_desc = {};
    clas_desc.op = ClusterAccelBuildOp::clas_from_triangles;
    clas_desc.args_buffer = BufferOffsetPair(args);
    clas_desc.args_stride = sizeof(OptixClusterAccelBuildInputTrianglesArgs);
    clas_desc.arg_count = 2;
    clas_desc.triangles_limits.max_arg_count = 2;
    clas_desc.triangles_limits.max_triangle_count_per_arg = 1;
    clas_desc.triangles_limits.max_vertex_count_per_arg = 3;
    clas_desc.triangles_limits.max_unique_sbt_index_count_per_arg = 1;

    ClusterAccelSizes clas_sizes = device->get_cluster_acceleration_structure_sizes(clas_desc);
    CHECK(clas_sizes.result_size > 0);

    ref<Buffer> clas_result = device->create_buffer({.size=clas_sizes.result_size, .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure, .default_state=ResourceState::acceleration_structure_build_output, .label="clas-out-batch"});
    ref<Buffer> clas_scratch = device->create_buffer({.size=clas_sizes.scratch_size, .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure, .default_state=ResourceState::acceleration_structure_build_output, .label="clas-scratch-batch"});

    ref<CommandEncoder> enc = device->create_command_encoder();
    enc->build_cluster_acceleration_structure(clas_desc, BufferOffsetPair(clas_scratch), BufferOffsetPair(clas_result));
    ref<CommandBuffer> cb = enc->finish();
    device->submit_command_buffer(cb.get());
    device->wait();

    uint64_t handles[2] = {};
    device->read_buffer_data(clas_result.get(), handles, sizeof(handles), 0);
    CHECK(handles[0] != 0);
    CHECK(handles[1] != 0);

    OptixClusterAccelBuildInputClustersArgs blas_args = {};
    blas_args.clusterHandlesCount = 2;
    blas_args.clusterHandlesBufferStrideInBytes = 8;
    blas_args.clusterHandlesBuffer = (CUdeviceptr)clas_result->device_address();
    ref<Buffer> blas_args_buf = device->create_buffer({.size=sizeof(blas_args), .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure_build_input, .default_state=ResourceState::acceleration_structure_build_output, .label="blas-args-batch"});
    blas_args_buf->set_data(&blas_args, sizeof(blas_args), 0);

    ClusterAccelBuildDesc blas_desc = {};
    blas_desc.op = ClusterAccelBuildOp::blas_from_clas;
    blas_desc.args_buffer = BufferOffsetPair(blas_args_buf);
    blas_desc.args_stride = sizeof(OptixClusterAccelBuildInputClustersArgs);
    blas_desc.arg_count = 1;
    blas_desc.clusters_limits.max_arg_count = 1;
    blas_desc.clusters_limits.max_total_cluster_count = 2;
    blas_desc.clusters_limits.max_cluster_count_per_arg = 2;

    ClusterAccelSizes blas_sizes = device->get_cluster_acceleration_structure_sizes(blas_desc);
    CHECK(blas_sizes.result_size > 0);

    ref<Buffer> blas_result = device->create_buffer({.size=blas_sizes.result_size, .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure, .default_state=ResourceState::acceleration_structure_build_output, .label="blas-out-batch"});
    ref<Buffer> blas_scratch = device->create_buffer({.size=blas_sizes.scratch_size, .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure, .default_state=ResourceState::acceleration_structure_build_output, .label="blas-scratch-batch"});

    enc = device->create_command_encoder();
    enc->build_cluster_acceleration_structure(blas_desc, BufferOffsetPair(blas_scratch), BufferOffsetPair(blas_result));
    cb = enc->finish();
    device->submit_command_buffer(cb.get());
    device->wait();

    uint64_t blas_qword = 0;
    device->read_buffer_data(blas_result.get(), &blas_qword, sizeof(blas_qword), 0);
    CHECK(blas_qword != 0);
}

// -----------------------------------------------------------------------------
// Limits and bad-args validation
// -----------------------------------------------------------------------------

static bool has_cluster_feature(ref<Device>& device)
{
    try { device = Device::create({.type = DeviceType::cuda}); } catch (...) { return false; }
    return device && device->info().optix_version >= 90000 && device->has_feature(Feature::cluster_acceleration_structure);
}

TEST_CASE("clas_missing_limits_should_fail_sizes")
{
    ref<Device> device; if (!has_cluster_feature(device)) return;

    ClusterAccelBuildDesc desc = {};
    desc.op = ClusterAccelBuildOp::clas_from_triangles;
    bool threw = false;
    try { (void)device->get_cluster_acceleration_structure_sizes(desc); }
    catch(...) { threw = true; }
    CHECK(threw);
}

TEST_CASE("clas_zero_required_limits_should_fail_sizes")
{
    ref<Device> device; if (!has_cluster_feature(device)) return;
    ClusterAccelBuildDesc desc = {};
    desc.op = ClusterAccelBuildOp::clas_from_triangles;
    desc.triangles_limits.max_arg_count = 0;
    desc.triangles_limits.max_triangle_count_per_arg = 1;
    desc.triangles_limits.max_vertex_count_per_arg = 1;
    desc.triangles_limits.max_unique_sbt_index_count_per_arg = 1;
    bool threw = false;
    try { (void)device->get_cluster_acceleration_structure_sizes(desc); }
    catch(...) { threw = true; }
    CHECK(threw);
}

TEST_CASE("blas_zero_required_limits_should_fail_sizes")
{
    ref<Device> device; if (!has_cluster_feature(device)) return;
    ClusterAccelBuildDesc desc = {};
    desc.op = ClusterAccelBuildOp::blas_from_clas;
    desc.clusters_limits.max_arg_count = 1;
    desc.clusters_limits.max_total_cluster_count = 0;
    desc.clusters_limits.max_cluster_count_per_arg = 1;
    bool threw = false;
    try { (void)device->get_cluster_acceleration_structure_sizes(desc); }
    catch(...) { threw = true; }
    CHECK(threw);
}

TEST_CASE("bad_args_fields_should_fail_build")
{
    ref<Device> device; if (!has_cluster_feature(device)) return;
    ClusterAccelBuildDesc desc = {};
    desc.op = ClusterAccelBuildOp::clas_from_triangles;
    desc.arg_count = 0;
    desc.args_stride = 0;
    desc.args_buffer = {};
    desc.triangles_limits.max_arg_count = 1;
    desc.triangles_limits.max_triangle_count_per_arg = 1;
    desc.triangles_limits.max_vertex_count_per_arg = 1;
    desc.triangles_limits.max_unique_sbt_index_count_per_arg = 1;

    ref<Buffer> scratch = device->create_buffer({.size=128, .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure, .default_state=ResourceState::acceleration_structure_build_output});
    ref<Buffer> result = device->create_buffer({.size=128, .memory_type=MemoryType::device_local, .usage=BufferUsage::acceleration_structure, .default_state=ResourceState::acceleration_structure_build_output});

    bool threw = false;
    try {
        ref<CommandEncoder> enc = device->create_command_encoder();
        enc->build_cluster_acceleration_structure(desc, BufferOffsetPair(scratch), BufferOffsetPair(result));
        (void)enc->finish();
    } catch(...) { threw = true; }
    CHECK(threw);
}

TEST_SUITE_END();

