#pragma once

#include <cstdint>

namespace cluster_abi {
    #define ABI_U32 uint32_t
    #define ABI_U64 uint64_t
    #include "cluster_accel_abi.common.inc"
    #undef ABI_U32
    #undef ABI_U64
}


