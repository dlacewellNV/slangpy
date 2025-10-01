### Vulkan NV cluster acceleration structure — summary (from vk.xml in build tree)

Source: build/linux-gcc/_deps/vulkan_headers-src/registry/vk.xml (auto-fetched by the build)

This summarizes the NV cluster acceleration structure extension entries found in the Vulkan headers registry. Names and shapes are subject to change (likely provisional/vendor-specific).

Key feature and properties
- VkPhysicalDeviceClusterAccelerationStructureFeaturesNV
  - field: clusterAccelerationStructure (enable/disable)
- VkPhysicalDeviceClusterAccelerationStructurePropertiesNV
  - limits: maxVerticesPerCluster, maxTrianglesPerCluster, various byte alignments (scratch/output/template), maxClusterGeometryIndex

Pipeline toggle
- VkRayTracingPipelineClusterAccelerationStructureCreateInfoNV
  - allowClusterAccelerationStructure: enable cluster geometry in pipeline

Geometry/cluster flags enums
- VkClusterAccelerationStructureGeometryFlagBitsNV
- VkClusterAccelerationStructureClusterFlagBitsNV
- VkClusterAccelerationStructureIndexFormatFlagBitsNV (format flags correspond to 1/2/4 byte indices)
- VkClusterAccelerationStructureAddressResolutionFlagBitsNV

Operation kinds and modes
- VkClusterAccelerationStructureOpTypeNV
  - BUILD_CLUSTERS_BOTTOM_LEVEL_NV
  - BUILD_TRIANGLE_CLUSTER_NV
  - BUILD_TRIANGLE_CLUSTER_TEMPLATE_NV
  - INSTANTIATE_TRIANGLE_CLUSTER_NV
- VkClusterAccelerationStructureOpModeNV
  - IMPLICIT_DESTINATIONS / EXPLICIT_DESTINATIONS / GET_SIZES

Host descriptor shape (mode-desc pattern)
- VkClusterAccelerationStructureInputInfoNV
  - fields: opType, opMode
  - union VkClusterAccelerationStructureOpInputNV:
    - pClustersBottomLevel (VkClusterAccelerationStructureClustersBottomLevelInputNV)
    - pTriangleClusters (VkClusterAccelerationStructureTriangleClusterInputNV)
    - pMoveObjects (VkClusterAccelerationStructureMoveObjectsInputNV)
- VkClusterAccelerationStructureCommandsInfoNV
  - embeds InputInfo + optional addressResolutionFlags

Triangle/cluster inputs (caps/upper-bounds)
- VkClusterAccelerationStructureTriangleClusterInputNV
  - maxClusterUniqueGeometryCount, maxClusterTriangleCount, maxClusterVertexCount
- VkClusterAccelerationStructureBuildTriangleClusterInfoNV / TemplateInfoNV
  - clusterID, clusterFlags, baseGeometryIndexAndGeometryFlags (per-cluster base)

Bottom-level GAS-from-clusters inputs
- VkClusterAccelerationStructureClustersBottomLevelInputNV
  - clusterReferences buffer (address/stride/count)

Notes on shape vs OptiX
- The Vulkan host API mirrors OptiX’s mode-desc design: an “op type + mode + union payload” for inputs, and separate structs expressing implicit/explicit/get-sizes behaviors.
- Per-primitive info and SBT index concepts also exist (via geometry index/flags and per-primitive buffers), with index formats aligning to 1/2/4 bytes.

Mapping guidance to slang-rhi
- RHI `ClusterAccelBuildDesc` (buildType/buildMode + modeDesc) maps naturally to Vulkan’s InputInfo/opMode + union.
  - buildType ↔ opType
  - buildMode ↔ opMode
  - modeDesc ↔ specific mode descriptor (implicit/explicit/get-sizes)
- Device ABI for per-primitive info (packed 32-bit sbtIndex/flags) matches OptiX; Vulkan aligns conceptually.
- Keep host descriptors backend-agnostic; only device “*Args” are ABI-bound.

Where to look in vk.xml
- Types: VkClusterAccelerationStructure*NV (features/properties/flags/enums)
- Inputs: VkClusterAccelerationStructure*InputNV, *InfoNV, *OpInputNV
- Modes: *BuildMode*, *CommandsInfoNV

Caveats
- This appears vendor-specific; names and exact fields may change.
- Treat as reference for backend mapping; do not copy verbatim into the common RHI.



