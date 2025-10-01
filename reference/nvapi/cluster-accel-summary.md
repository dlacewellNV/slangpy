### NVAPI D3D12 cluster acceleration structure — summary

Source: reference/nvapi/nvapi.h and related headers (vendor SDK)

Caps and pipeline toggles
- NVAPI_D3D12_RAYTRACING_CLUSTER_OPERATIONS_CAPS
  - STANDARD flag for availability
- NVAPI_D3D12_PIPELINE_CREATION_STATE_FLAGS_ENABLE_CLUSTER_SUPPORT
  - Enable cluster support on pipeline creation

Operation types and modes (multi-indirect)
- NVAPI_D3D12_RAYTRACING_MULTI_INDIRECT_CLUSTER_OPERATION_TYPE
  - MOVE_CLUSTER_OBJECT
  - BUILD_BLAS_FROM_CLAS
  - BUILD_CLAS_FROM_TRIANGLES
  - BUILD_CLUSTER_TEMPLATES_FROM_TRIANGLES
- Modes mirror OptiX/Vulkan (implicit/explicit/get-sizes style via descs)

Flags
- NVAPI_D3D12_RAYTRACING_MULTI_INDIRECT_CLUSTER_OPERATION_FLAGS
  - FAST_TRACE / FAST_BUILD, NO_OVERLAP, ALLOW_OMM (reserved for MVP)
- NVAPI_D3D12_RAYTRACING_MULTI_INDIRECT_CLUSTER_OPERATION_CLUSTER_FLAGS
  - ALLOW_DISABLE_OMMS (reserved for MVP)
- Geometry flags mirror D3D12/VK style (opaque, no-duplicate-anyhit, cull disable)

Alignment/limits
- NVAPI_D3D12_RAYTRACING_CLUSTER_TEMPLATE_BYTE_ALIGNMENT = 32
- NVAPI_D3D12_RAYTRACING_CLUSTER_TEMPLATE_BOUNDS_BYTE_ALIGNMENT = 32
- NVAPI_D3D12_RAYTRACING_MAXIMUM_GEOMETRY_INDEX = 16,777,215

Shader intrinsics (HLSL)
- NvRtGetClusterID(), NvRtGetCandidateClusterID(), NvRtGetCommittedClusterID()
- Hit object helpers expose cluster ID as well

Mapping guidance to slang-rhi
- RHI build desc and mode-desc approach maps to NVAPI’s multi-indirect operation inputs.
- Device ABI for per-primitive info (sbtIndex + flags packed) remains consistent with OptiX and conceptually matches NVAPI.
- Keep OMM-related flags reserved and zero in MVP.

Caveats
- Vendor API, subject to change; keep RHI headers vendor-agnostic and isolate mappings in backends.



