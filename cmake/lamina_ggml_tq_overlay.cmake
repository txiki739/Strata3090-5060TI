# cmake/lamina_ggml_tq_overlay.cmake - MMQ prompt kernels for Gyro-L's trellis types (TQP5, TQP6, TQP7) on the pinned llama.cpp.
#
# The agentionai llama.cpp fork has these types in its ggml (ggml.h enum, ggml-cuda/tq.cuh, a tile loader, MMQ configs and
# routing, template instances).  Upstream does not.  This script writes, at configure time, a small OVERLAY of the pinned ggml-cuda
# headers into the build tree: copies of ggml.h, mmq.cuh, mmq-load-tiles.cuh and the mmq-config-*.cuh with the trellis types added
# the way IQ2_XS is handled (the same Q8_0_16 tile layout, dot kernels and configs: the fork's own numbers for them are copies of
# IQ2_XS's), plus tq.cuh / tq-mmq-tiles.cuh from third_party/ggml-tq and three template instances.  Only the lamina_mmq target sees
# the overlay (it comes first on its include path); every other type's instances are compiled from the untouched pinned files.
#
# Every edit is anchored on pinned text; if an anchor is missing (another llama.cpp), the script says so and the trellis MMQ stays
# off (LAMINA_TQ_OVERLAY_OK=OFF): the engine then reads trellis prompts through the FP16 path.
#
#   lamina_ggml_tq_overlay(<llama.cpp dir> <out dir>)  ->  LAMINA_TQ_OVERLAY_OK (parent scope)

function(lamina_ggml_tq_overlay LLAMA OUT)
  set(LAMINA_TQ_OVERLAY_OK OFF PARENT_SCOPE)
  set(_cu "${LLAMA}/ggml/src/ggml-cuda")
  set(_inc "${LLAMA}/ggml/include")
  set(_tq "${CMAKE_SOURCE_DIR}/third_party/ggml-tq")
  foreach(_f "${_inc}/ggml.h" "${_cu}/mmq.cuh" "${_cu}/mmq-load-tiles.cuh" "${_cu}/common.cuh")
    if(NOT EXISTS "${_f}")
      message(STATUS "trellis MMQ overlay: ${_f} is missing: off")
      return()
    endif()
  endforeach()

  # ---- helpers: read a file with LF line ends; insert text before / after the single occurrence of an anchor ----------
  macro(_tq_read VAR PATH)
    file(READ "${PATH}" ${VAR})
    string(REPLACE "\r\n" "\n" ${VAR} "${${VAR}}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${PATH}")
  endmacro()
  macro(_tq_edit VAR WHERE ANCHOR TEXT)   # WHERE = BEFORE | AFTER; the anchor must occur exactly once
    string(FIND "${${VAR}}" "${ANCHOR}" _p)
    if(_p LESS 0)
      set(_tq_fail "anchor not found: ${ANCHOR}")
    else()
      string(LENGTH "${ANCHOR}" _alen)
      math(EXPR _after "${_p} + ${_alen}")
      string(SUBSTRING "${${VAR}}" ${_after} -1 _tail)
      string(FIND "${_tail}" "${ANCHOR}" _p2)
      if(NOT _p2 LESS 0)
        set(_tq_fail "anchor is not unique: ${ANCHOR}")
      else()
        string(SUBSTRING "${${VAR}}" 0 ${_p} _head)
        if("${WHERE}" STREQUAL "AFTER")
          set(${VAR} "${_head}${ANCHOR}${TEXT}${_tail}")
        else()
          set(${VAR} "${_head}${TEXT}${ANCHOR}${_tail}")
        endif()
      endif()
    endif()
  endmacro()
  set(_tq_fail "")

  file(MAKE_DIRECTORY "${OUT}/include" "${OUT}/template-instances")

  # ---- the whole header set of ggml-cuda and ggml/include is copied: quote includes resolve next to the including file,
  # so a header left in the pinned directory would pull the pinned mmq.cuh / ggml.h back in beside the patched ones -------
  file(GLOB _all_cu "${_cu}/*.cuh" "${_cu}/*.h")
  foreach(_f IN LISTS _all_cu)
    get_filename_component(_n "${_f}" NAME)
    configure_file("${_f}" "${OUT}/${_n}" COPYONLY)
  endforeach()
  configure_file("${_cu}/quantize.cu" "${OUT}/quantize.cu" COPYONLY)   # compiled from here: it includes mmq.cuh
  if(IS_DIRECTORY "${_cu}/vendors")
    file(COPY "${_cu}/vendors" DESTINATION "${OUT}")
  endif()
  file(GLOB _all_inc "${_inc}/*.h")
  foreach(_f IN LISTS _all_inc)
    get_filename_component(_n "${_f}" NAME)
    configure_file("${_f}" "${OUT}/include/${_n}" COPYONLY)
  endforeach()

  # ---- ggml.h: the enumerators (values above GGML_TYPE_COUNT: ggml's own tables never see them) --------------------------
  _tq_read(_h "${_inc}/ggml.h")
  _tq_edit(_h BEFORE "        GGML_TYPE_COUNT   = 43,"
    "        GGML_TYPE_TQP5    = 147, // Gyro-L's pair trellis (the agentionai fork's ids; this engine's overlay)
        GGML_TYPE_TQP6    = 148,
        GGML_TYPE_TQP7    = 149,
")
  file(WRITE "${OUT}/include/ggml.h" "${_h}")

  # ---- mmq.cuh: routing of the trellis types as IQ2_XS ----------------------------------------------------------------
  _tq_read(_m "${_cu}/mmq.cuh")
  _tq_edit(_m AFTER "        case GGML_TYPE_IQ2_XXS:\n        case GGML_TYPE_IQ2_XS:\n"
    "        case GGML_TYPE_TQP5:\n        case GGML_TYPE_TQP6:\n        case GGML_TYPE_TQP7:\n")
  _tq_edit(_m AFTER "        case GGML_TYPE_IQ2_XS:  return MMQ_DP4A_TXS_Q8_0_16;\n"
    "        case GGML_TYPE_TQP5:    return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQP6:    return MMQ_DP4A_TXS_Q8_0_16;\n        case GGML_TYPE_TQP7:    return MMQ_DP4A_TXS_Q8_0_16;\n")
  _tq_edit(_m BEFORE "            case GGML_TYPE_IQ3_XXS:\n                return ggml_cuda_mmq_util_funcs(\n                    VDR_IQ3_XXS_Q8_1_MMQ,"
    "            case GGML_TYPE_TQP5:\n            case GGML_TYPE_TQP6:\n            case GGML_TYPE_TQP7:\n                return ggml_cuda_mmq_util_funcs(\n                    VDR_IQ2_XS_Q8_1_MMQ,\n                    ggml_cuda_mmq_load_tiles_tq<type, J, fallback>,\n                    ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a<type, J, fallback>,\n                    ggml_cuda_mmq_write_back_dp4a<type, J, fallback>);\n")
  _tq_edit(_m BEFORE "        case GGML_TYPE_IQ3_XXS:\n            return ggml_cuda_mmq_util_funcs(\n                -1,\n                ggml_cuda_mmq_load_tiles_iq3_xxs<type, J, fallback>,"
    "        case GGML_TYPE_TQP5:\n        case GGML_TYPE_TQP6:\n        case GGML_TYPE_TQP7:\n            return ggml_cuda_mmq_util_funcs(\n                -1,\n                ggml_cuda_mmq_load_tiles_tq<type, J, fallback>,\n                ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma<type, J, fallback>,\n                ggml_cuda_mmq_write_back_mma<type, J, fallback>);\n")
  _tq_edit(_m AFTER "extern DECL_MMQ_CASE(GGML_TYPE_IQ4_XS);\n"
    "extern DECL_MMQ_CASE(GGML_TYPE_TQP5);\nextern DECL_MMQ_CASE(GGML_TYPE_TQP6);\nextern DECL_MMQ_CASE(GGML_TYPE_TQP7);\n")
  file(WRITE "${OUT}/mmq.cuh" "${_m}")

  # ---- mmq-load-tiles.cuh: the loader, at the end -------------------------------------------------------------------
  _tq_read(_l "${_cu}/mmq-load-tiles.cuh")
  file(WRITE "${OUT}/mmq-load-tiles.cuh" "${_l}\n#include \"tq-mmq-tiles.cuh\"\n")
  foreach(_n tq.cuh tq-mmq-tiles.cuh ggml-common-tq.h tq_lut_i8.cuh)
    configure_file("${_tq}/${_n}" "${OUT}/${_n}" COPYONLY)
  endforeach()

  # ---- mmq-config-*.cuh: every IQ2_XS CASE line gets one copy per trellis type ---------------------------------------
  file(GLOB _cfgs "${_cu}/mmq-config-*.cuh")
  foreach(_c ${_cfgs})
    get_filename_component(_cn "${_c}" NAME)
    _tq_read(_t "${_c}")
    string(REPLACE ";" "<SEMI>" _t "${_t}")   # the CASE lines hold semicolons; a CMake list would split them
    string(REGEX MATCHALL "[^\n]*CASE\\(GGML_TYPE_IQ2_XS,[^\n]*\n" _lines "${_t}")
    list(REMOVE_DUPLICATES _lines)
    foreach(_ln IN LISTS _lines)
      set(_add "")
      foreach(_ty TQP5 TQP6 TQP7)
        string(REPLACE "GGML_TYPE_IQ2_XS," "GGML_TYPE_${_ty}," _x "${_ln}")
        string(APPEND _add "${_x}")
      endforeach()
      _tq_edit(_t AFTER "${_ln}" "${_add}")
    endforeach()
    string(REPLACE "<SEMI>" ";" _t "${_t}")
    file(WRITE "${OUT}/${_cn}" "${_t}")
  endforeach()

  # ---- template instances -------------------------------------------------------------------------------------------
  foreach(_ty tqp5 tqp6 tqp7)
    string(TOUPPER "${_ty}" _TY)
    file(WRITE "${OUT}/template-instances/mmq-instance-${_ty}.cu"
         "// the trellis overlay (cmake/lamina_ggml_tq_overlay.cmake)\n\n#include \"../mmq.cuh\"\n\nDECL_MMQ_CASE(GGML_TYPE_${_TY});\n")
  endforeach()

  if(_tq_fail)
    message(STATUS "trellis MMQ overlay off: ${_tq_fail}")
    return()
  endif()
  set(LAMINA_TQ_OVERLAY_OK ON PARENT_SCOPE)
endfunction()
