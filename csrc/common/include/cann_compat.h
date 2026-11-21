#pragma once

#include <stdint.h>

// B080 op_common/log/log.h stopped exposing the unqualified OP module id used
// by inherited ops-transformer tiling/error headers. Include the CANN log type
// header early so OP still comes from the active CANN version.
#if defined(__has_include)
#if __has_include("base/log_types.h")
#include "base/log_types.h"
#elif __has_include("toolchain/log_types.h")
#include "toolchain/log_types.h"
#endif
#endif

#if !defined(LOG_TYPES_H_) && !defined(OP)
#define OP 63
#endif

#if defined(LOG_CPP) && !defined(DLOG_PUB_H_)
#ifdef __cplusplus
extern "C" {
#endif
int32_t CheckLogLevel(int32_t moduleId, int32_t logLevel);
void DlogRecord(int32_t moduleId, int32_t level, const char *fmt, ...);
#ifdef __cplusplus
}
#endif
#define DLOG_PUB_H_
#endif

// torch_npu 2.13 dev headers (torch_npu/csrc/core/npu/interface/AclInterface.h)
// declare aclmdlRICond* APIs whose typedefs live in the ACL headers vendored
// by torch_npu (third_party/acl/inc/acl/acl_rt.h) but not in the pinned CANN
// 9.1.0-beta.1. Redeclare them exactly as vendored (typedef redeclaration of
// the same types is well-formed, so this stays correct even when the vendored
// header is also included); vllm-ascend never calls these APIs.
#if defined(__cplusplus) && !defined(VLLM_ASCEND_ACLMDL_RI_COND_COMPAT)
#define VLLM_ASCEND_ACLMDL_RI_COND_COMPAT
typedef void *aclmdlRICondHandle;
typedef struct tagAclmdlRICondTaskParams aclmdlRICondTaskParams;
#endif
