#include "sgl/sgl.h"
#include "sgl/core/platform.h"
#include "sgl/device/device.h"
#include "sgl/device/types.h"
#include "sgl/device/shader.h"
#include "sgl/device/pipeline.h"
#include "sgl/device/command.h"
#include "sgl/device/raytracing.h"
#include "sgl/device/shader_cursor.h"
#include "sgl/device/shader_object.h"
#include "sgl/utils/tev.h"

#include <slang-rhi/cluster_accel_abi_host.h>

#include <filesystem>

static const std::filesystem::path EXAMPLE_DIR(SGL_EXAMPLE_DIR);

using namespace sgl;

#ifndef SLANG_RHI_SHADERS_DIR
#error "SLANG_RHI_SHADERS_DIR not defined. Ensure CMake exports it from slang-rhi."
#endif

// This example demonstrates cluster acceleration structures:
//
// 1. Create geometry for two separate triangle strips (sharing index topology)
// 2. Build 2 CLAS (cluster acceleration structures) from triangles using device-written args
// 3. Build 1 BLAS from the 2 CLAS handles
// 4. Build TLAS from the BLAS
// 5. Render using a ray tracing pipeline with clusters enabled
//
// Key features demonstrated:
// - Device-written CLAS args (compute shader writes args, barrier, then build)
// - Implicit build mode (driver allocates within result buffer, returns handles)
// - Per-cluster vertex base offsets (shared index buffer, different vertex ranges)
// - Handle-based BLAS construction from CLAS

int main()
{
    sgl::static_init();

    const std::filesystem::path SHARED_SHADERS_DIR(SLANG_RHI_SHADERS_DIR);
    ref<Device> device = Device::create({
        .type = DeviceType::cuda,
        .enable_debug_layers = true,
        .compiler_options = {.include_paths = {EXAMPLE_DIR, SHARED_SHADERS_DIR}},
    });

    if (!device->has_feature(Feature::ray_tracing)) {
        log_info("Ray tracing not supported. Skipping.");
        return 0;
    }

    if (!device->has_feature(Feature::cluster_acceleration_structure)) {
        log_info("Cluster acceleration structure not supported. Skipping.");
        return 0;
    }

    // --- Geometry: 1x4 grid (8 triangles per strip) ---
    struct Float3 { float x, y, z; };
    constexpr uint32_t kGridW = 4;
    constexpr uint32_t kGridH = 1;
    constexpr uint32_t triCount = kGridW * kGridH * 2;
    constexpr uint32_t vertW = kGridW + 1;
    constexpr uint32_t vertH = kGridH + 1;
    constexpr uint32_t vertCount = vertW * vertH;

    std::vector<Float3> vertices;
    vertices.reserve(vertCount);
    for (uint32_t j = 0; j < vertH; ++j) {
        for (uint32_t i = 0; i < vertW; ++i) {
            float u = float(i) / float(kGridW);
            float v = float(j) / float(kGridH);
            float x = -0.8f + u * 1.6f;
            float y = -0.2f + v * 0.4f;
            vertices.push_back({x, y, 0.0f});
        }
    }

    std::vector<uint32_t> indices;
    indices.reserve(triCount * 3);
    for (uint32_t j = 0; j < kGridH; ++j) {
        for (uint32_t i = 0; i < kGridW; ++i) {
            uint32_t i0 = j * vertW + i;
            uint32_t i1 = j * vertW + (i + 1);
            uint32_t i2 = (j + 1) * vertW + i;
            uint32_t i3 = (j + 1) * vertW + (i + 1);
            // Two triangles per cell
            indices.push_back(i0); indices.push_back(i1); indices.push_back(i3);
            indices.push_back(i3); indices.push_back(i2); indices.push_back(i0);
        }
    }

    // Append a second strip above the first, with a small vertical gap (vertices only; reuse indices)
    constexpr float kStripGap = 0.25f;
    for (uint32_t i = 0; i < vertCount; ++i) {
        Float3 vtx = vertices[i];
        vtx.y += 0.4f + kStripGap; // shift up by strip height plus gap
        vertices.push_back(vtx);
    }

    ref<Buffer> vertexBuffer = device->create_buffer({
        .usage = BufferUsage::acceleration_structure_build_input,
        .label = "vertex_buffer",
        .data = vertices.data(),
        .data_size = vertices.size() * sizeof(Float3),
    });
    ref<Buffer> indexBuffer = device->create_buffer({
        .usage = BufferUsage::acceleration_structure_build_input,
        .label = "index_buffer",
        .data = indices.data(),
        .data_size = indices.size() * sizeof(uint32_t),
    });

	// --- Build a single template from triangles, then instantiate twice (implicit mode) ---

	// 1) TemplatesFromTriangles: host-filled TrianglesArgs (topology is shared)
	// Note: For simplicity, we fill args on the host. For device-driven scenes with many
	// clusters, use a compute shader to write args (see slang-rhi cluster tests).
	constexpr uint32_t clusterCount = 2;
	cluster_abi::TrianglesArgs trianglesArgs = cluster_abi::makeTrianglesArgs(
		/*clusterId*/0,
		/*triangleCount*/triCount,
		/*vertexCount*/vertCount,
		/*indexBuffer*/(uint64_t)indexBuffer->device_address(),
		/*vertexBuffer*/(uint64_t)vertexBuffer->device_address(),
		/*vertexStrideBytes*/(uint32_t)sizeof(Float3),
		/*indexFormat*/4 // 32-bit indices
	);
	ref<Buffer> templatesTrianglesArgs = device->create_buffer({
		.usage = BufferUsage::acceleration_structure_build_input,
		.label = "templates_from_triangles_args",
		.data = &trianglesArgs,
		.data_size = sizeof(trianglesArgs),
	});

	ClusterAccelBuildDesc templatesDesc{};
	templatesDesc.op = ClusterAccelBuildOp::templates_from_triangles;
	templatesDesc.args_buffer = {templatesTrianglesArgs, 0};
	templatesDesc.args_stride = sizeof(cluster_abi::TrianglesArgs);
	templatesDesc.arg_count = 1;
	templatesDesc.limits.limits_triangles.max_arg_count = 1;
	templatesDesc.limits.limits_triangles.max_triangle_count_per_arg = triCount;
	templatesDesc.limits.limits_triangles.max_vertex_count_per_arg = vertCount;
	templatesDesc.limits.limits_triangles.max_unique_sbt_index_count_per_arg = 1;

	ClusterAccelSizes templateSizes = device->get_cluster_acceleration_structure_sizes(templatesDesc);
	log_info("Template sizes: result={} scratch={}", templateSizes.result_size, templateSizes.scratch_size);

	ref<Buffer> templateHandles = device->create_buffer({
		.size = 8, // one handle
		.usage = BufferUsage::unordered_access,
		.label = "template_handles",
	});
	ref<Buffer> templateData = device->create_buffer({
		.size = templateSizes.result_size,
		.usage = BufferUsage::acceleration_structure,
		.label = "template_data",
	});
	ref<Buffer> templateScratch = device->create_buffer({
		.size = templateSizes.scratch_size,
		.usage = BufferUsage::unordered_access,
		.label = "template_scratch",
	});

	{
		auto enc = device->create_command_encoder();

		// Build the template (implicit mode)
		templatesDesc.mode = ClusterAccelBuildDesc::BuildMode::implicit;
		templatesDesc.implicit.output_handles_buffer = templateHandles->device_address();
		templatesDesc.implicit.output_handles_stride_in_bytes = 0; // 0 -> 8
		templatesDesc.implicit.output_buffer = templateData->device_address();
		templatesDesc.implicit.output_buffer_size_in_bytes = templateData->size();
		templatesDesc.implicit.temp_buffer = templateScratch->device_address();
		templatesDesc.implicit.temp_buffer_size_in_bytes = templateScratch->size();

		enc->build_cluster_acceleration_structure(templatesDesc);
		device->submit_command_buffer(enc->finish());
	}

	uint64_t templateHandle = 0;
	device->read_buffer_data(templateHandles, &templateHandle, sizeof(templateHandle), 0);
	if (templateHandle == 0) {
		log_error("TemplatesFromTriangles failed: template handle is zero");
		return 1;
	}

	// 2) CLASFromTemplates: instantiate the template twice with per-instance vertex bases
	cluster_abi::TemplatesArgs templateInstantiationArgs[clusterCount] = {};
	uint64_t vertexBase = (uint64_t)vertexBuffer->device_address();
	uint32_t vertexStride = (uint32_t)sizeof(Float3);
	for (uint32_t i = 0; i < clusterCount; ++i) {
		uint64_t vertexAddr = vertexBase + uint64_t(i) * uint64_t(vertCount) * uint64_t(vertexStride);
		templateInstantiationArgs[i] = cluster_abi::makeTemplatesArgs(
			/*clusterTemplate*/templateHandle,
			/*vertexBuffer*/vertexAddr,
			/*vertexStrideInBytes*/vertexStride,
			/*clusterIdOffset*/i,
			/*sbtIndexOffset*/0);
	}
	ref<Buffer> clasFromTemplatesArgs = device->create_buffer({
		.usage = BufferUsage::acceleration_structure_build_input,
		.label = "clas_from_templates_args",
		.data = &templateInstantiationArgs[0],
		.data_size = sizeof(templateInstantiationArgs),
	});

	ClusterAccelBuildDesc clasFromTemplatesDesc{};
	clasFromTemplatesDesc.op = ClusterAccelBuildOp::clas_from_templates;
	clasFromTemplatesDesc.args_buffer = {clasFromTemplatesArgs, 0};
	clasFromTemplatesDesc.args_stride = sizeof(cluster_abi::TemplatesArgs);
	clasFromTemplatesDesc.arg_count = clusterCount;
	// Reuse triangle limits per API
	clasFromTemplatesDesc.limits.limits_triangles.max_arg_count = clusterCount;
	clasFromTemplatesDesc.limits.limits_triangles.max_triangle_count_per_arg = triCount;
	clasFromTemplatesDesc.limits.limits_triangles.max_vertex_count_per_arg = vertCount;
	clasFromTemplatesDesc.limits.limits_triangles.max_unique_sbt_index_count_per_arg = 1;

	ClusterAccelSizes clasSizes = device->get_cluster_acceleration_structure_sizes(clasFromTemplatesDesc);
	log_info("CLAS sizes: result={} scratch={}", clasSizes.result_size, clasSizes.scratch_size);

	// Allocate handles buffer (8 bytes per cluster) and result buffer for CLAS data
	ref<Buffer> clasHandles = device->create_buffer({
		.size = uint64_t(clusterCount) * 8u,
		.usage = BufferUsage::unordered_access,
		.label = "clas_handles",
	});
	ref<Buffer> clasData = device->create_buffer({
		.size = clasSizes.result_size,
		.usage = BufferUsage::acceleration_structure,
		.label = "clas_data",
	});
	ref<Buffer> clasScratch = device->create_buffer({
		.size = clasSizes.scratch_size,
		.usage = BufferUsage::unordered_access,
		.label = "clas_scratch",
	});

	{
		auto enc = device->create_command_encoder();
		clasFromTemplatesDesc.mode = ClusterAccelBuildDesc::BuildMode::implicit;
		clasFromTemplatesDesc.implicit.output_handles_buffer = clasHandles->device_address();
		clasFromTemplatesDesc.implicit.output_handles_stride_in_bytes = 0; // 0 -> 8
		clasFromTemplatesDesc.implicit.output_buffer = clasData->device_address();
		clasFromTemplatesDesc.implicit.output_buffer_size_in_bytes = clasData->size();
		clasFromTemplatesDesc.implicit.temp_buffer = clasScratch->device_address();
		clasFromTemplatesDesc.implicit.temp_buffer_size_in_bytes = clasScratch->size();

		enc->build_cluster_acceleration_structure(clasFromTemplatesDesc);
		device->submit_command_buffer(enc->finish());
	}

	// Verify CLAS build succeeded by checking handles are non-zero.
	uint64_t handles[2] = {0, 0};
	device->read_buffer_data(clasHandles, handles, sizeof(handles), 0);
	log_info("CLAS handles[0] = 0x{:016x}, CLAS handles[1] = 0x{:016x}", handles[0], handles[1]);
	if (handles[0] == 0 || handles[1] == 0) {
		log_error("CLAS build failed: one or more handles are zero (driver did not write valid addresses)");
		return 1;
	}

    // --- Build BLAS from CLAS handles ---
    cluster_abi::ClustersArgs clArgs = cluster_abi::makeClustersArgs(
        clusterCount,
        (uint64_t)clasHandles->device_address(),
        /*strideBytes*/8
    );
    ref<Buffer> blasArgsBuf = device->create_buffer({
        .usage = BufferUsage::acceleration_structure_build_input,
        .label = "blas_from_clas_args",
        .data = &clArgs,
        .data_size = sizeof(clArgs),
    });

    ClusterAccelBuildDesc blasDesc{};
    blasDesc.op = ClusterAccelBuildOp::blas_from_clas;
    blasDesc.args_buffer = {blasArgsBuf, 0};
    blasDesc.args_stride = sizeof(cluster_abi::ClustersArgs);
    blasDesc.arg_count = 1;
    blasDesc.limits.limits_clusters.max_arg_count = 1;
    blasDesc.limits.limits_clusters.max_total_cluster_count = clusterCount;
    blasDesc.limits.limits_clusters.max_cluster_count_per_arg = clusterCount;

    ClusterAccelSizes blasSizes = device->get_cluster_acceleration_structure_sizes(blasDesc);
    log_info("BLAS sizes: result={} scratch={}", blasSizes.result_size, blasSizes.scratch_size);

    // Same buffer pattern: handles, acceleration structure data, and scratch
    ref<Buffer> blasHandles = device->create_buffer({
        .size = 8, // one handle
        .usage = BufferUsage::unordered_access,
        .label = "blas_handles",
    });
    ref<Buffer> blasData = device->create_buffer({
        .size = blasSizes.result_size,
        .usage = BufferUsage::acceleration_structure,
        .label = "blas_data",
    });
    ref<Buffer> blasScratch = device->create_buffer({
        .size = blasSizes.scratch_size,
        .usage = BufferUsage::unordered_access,
        .label = "blas_scratch",
    });

    {
        auto enc = device->create_command_encoder();
        // Build BLAS from CLAS (implicit mode) — set required buffers in desc
        blasDesc.mode = ClusterAccelBuildDesc::BuildMode::implicit;
        blasDesc.implicit.output_handles_buffer = blasHandles->device_address();
        blasDesc.implicit.output_handles_stride_in_bytes = 0;
        blasDesc.implicit.output_buffer = blasData->device_address();
        blasDesc.implicit.output_buffer_size_in_bytes = blasData->size();
        blasDesc.implicit.temp_buffer = blasScratch->device_address();
        blasDesc.implicit.temp_buffer_size_in_bytes = blasScratch->size();

        enc->build_cluster_acceleration_structure(blasDesc);
        device->submit_command_buffer(enc->finish());
    }

    // Verify BLAS build succeeded
    uint64_t blasHandle = 0;
    device->read_buffer_data(blasHandles, &blasHandle, sizeof(blasHandle), 0);
    log_info("BLAS handle = 0x{:016x}", blasHandle);
    if (blasHandle == 0) {
        log_error("BLAS build failed: handle is zero (driver did not write valid address)");
        return 1;
    }

    // --- TLAS from BLAS ---
    ref<AccelerationStructureInstanceList> instanceList = device->create_acceleration_structure_instance_list(1);
    instanceList->write(0, AccelerationStructureInstanceDesc{
        .transform = float3x4::identity(),
        .instance_id = 0,
        .instance_mask = 0xff,
        .instance_contribution_to_hit_group_index = 0,
        .flags = AccelerationStructureInstanceFlags::none,
        .acceleration_structure = AccelerationStructureHandle{blasHandle},
    });

    AccelerationStructureBuildDesc tlasBuildDesc{
        .inputs = {instanceList->build_input_instances()},
    };
    AccelerationStructureSizes tlasSizes = device->get_acceleration_structure_sizes(tlasBuildDesc);
    ref<Buffer> tlasScratch = device->create_buffer({
        .size = tlasSizes.scratch_size,
        .usage = BufferUsage::unordered_access,
        .label = "tlas_scratch",
    });
    ref<AccelerationStructure> tlas = device->create_acceleration_structure({
        .size = tlasSizes.acceleration_structure_size,
        .label = "tlas",
    });
    {
        auto enc = device->create_command_encoder();
        enc->build_acceleration_structure(tlasBuildDesc, tlas, nullptr, tlasScratch);
        device->submit_command_buffer(enc->finish());
    }

    // --- Output texture ---
    const uint32_t W = 512, H = 512;
    ref<Texture> renderTexture = device->create_texture({
        .format = Format::rgba32_float,
        .width = W,
        .height = H,
        .usage = TextureUsage::unordered_access,
        .label = "render_texture",
    });

    // --- Pipeline + shader table ---
    auto program = device->load_program("raytracing_pipeline_clusters.slang", {"ray_gen", "miss", "closest_hit"});
    RayTracingPipelineDesc pdesc{};
    pdesc.program = program;
    pdesc.hit_groups = {{
        .hit_group_name = "hit_group",
        .closest_hit_entry_point = "closest_hit",
    }};
    pdesc.max_recursion = 1;
    pdesc.max_ray_payload_size = 16;
    pdesc.flags = RayTracingPipelineFlags::enable_clusters;
    auto pipeline = device->create_ray_tracing_pipeline(pdesc);
    auto shaderTable = device->create_shader_table({
        .program = program,
        .ray_gen_entry_points = {"ray_gen"},
        .miss_entry_points = {"miss"},
        .hit_group_names = {"hit_group"},
    });

    // --- Dispatch rays ---
    {
        auto enc = device->create_command_encoder();
        auto pass = enc->begin_ray_tracing_pass();
        auto shaderObject = pass->bind_pipeline(pipeline, shaderTable);
        ShaderCursor cursor(shaderObject);
        cursor["tlas"] = tlas;
        cursor["render_texture"] = renderTexture;
        pass->dispatch_rays(0, uint3{W, H, 1});
        pass->end();
        device->submit_command_buffer(enc->finish());
    }

    // --- Send to Tev ---
    tev::show(renderTexture, "raytracing_pipeline_clusters");

    return 0;
}


