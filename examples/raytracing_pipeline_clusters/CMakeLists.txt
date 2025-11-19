sgl_add_example(raytracing_pipeline_clusters)
target_sources(raytracing_pipeline_clusters PRIVATE raytracing_pipeline_clusters.cpp)

file(TO_CMAKE_PATH "${slang-rhi_SOURCE_DIR}/include" SLANG_RHI_INCLUDE_DIR)
target_compile_definitions(raytracing_pipeline_clusters PRIVATE SLANG_RHI_INCLUDE_DIR="${SLANG_RHI_INCLUDE_DIR}")

