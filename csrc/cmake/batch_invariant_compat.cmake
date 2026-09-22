# Compatibility for operator sources that select a per-SoC kernel entry file.
set(kernel_src_list CACHE INTERNAL "kernel source" FORCE)

function(_bi_get_op_type_from_binary_json binary_json op_type_out)
  file(READ "${binary_json}" binary_json_content)
  string(JSON op_type ERROR_VARIABLE json_error GET "${binary_json_content}" op_type)
  if(json_error)
    set(op_type "")
  endif()
  set(${op_type_out} "${op_type}" PARENT_SCOPE)
endfunction()

function(add_kernel_sources)
  set(one_value_args KERNEL_SRC SIMPLIFIED_KEY AUTO_SYNC IMPL_MODE)
  set(multi_value_args COMPUTE_UNITS OPTIONS)
  cmake_parse_arguments(KERNEL "" "${one_value_args}" "${multi_value_args}" ${ARGN})

  set(matched_compute_unit "")
  foreach(compute_unit ${ASCEND_COMPUTE_UNIT})
    if(NOT KERNEL_COMPUTE_UNITS OR compute_unit IN_LIST KERNEL_COMPUTE_UNITS)
      set(matched_compute_unit "${compute_unit}")
      break()
    endif()
  endforeach()
  if(NOT matched_compute_unit)
    return()
  endif()

  get_filename_component(op_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
  get_filename_component(op_name "${op_dir}" NAME)
  set(binary_json "${op_dir}/op_host/config/${matched_compute_unit}/${op_name}_binary.json")
  if(NOT EXISTS "${binary_json}")
    message(FATAL_ERROR "Missing ${matched_compute_unit} binary config for ${op_name}")
  endif()
  _bi_get_op_type_from_binary_json("${binary_json}" op_type)
  if(NOT op_type)
    message(FATAL_ERROR "Missing op_type in ${binary_json}")
  endif()

  if(KERNEL_KERNEL_SRC)
    string(REGEX REPLACE "\\.cpp$" "" kernel_src "${KERNEL_KERNEL_SRC}")
  else()
    set(kernel_src "${op_name}")
  endif()
  list(APPEND kernel_src_list "${op_type} ${matched_compute_unit} ${kernel_src}")
  set(kernel_src_list "${kernel_src_list}" CACHE INTERNAL "kernel source" FORCE)
endfunction()

# ops-batchinvariant normally builds as a standalone operator project. Reuse
# its framework support in the transformer package while preserving one host
# and one op-api library for all native and BI operators.
function(batch_invariant_prepare root)
  # BUILD_WITH_3_8_PACKAGE omits opsbase from the transformer host libraries,
  # but the imported BI host code uses Ops::Base conversion helpers from it.
  # Link it explicitly so the generated tiling library can be loaded by opc.
  if(TARGET opsbase)
    if(TARGET cust_opmaster)
      target_link_libraries(cust_opmaster PRIVATE opsbase)
    endif()
    if(TARGET cust_opapi)
      target_link_libraries(cust_opapi PRIVATE opsbase)
    endif()
    if(TARGET cust_proto)
      target_link_libraries(cust_proto PRIVATE opsbase)
    endif()
  endif()

  file(GLOB bi_common_tiling_sources
       "${root}/common/src/*.cpp"
       "${root}/common/src/op_host/*.cpp")
  if(TARGET ${OPHOST_NAME}_tiling_obj)
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${bi_common_tiling_sources})
    target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE
                               NN_ENABLE_DLOPEN_LEGACY)
  endif()

  if(TARGET ${OPHOST_NAME}_opapi_obj)
    target_sources(${OPHOST_NAME}_opapi_obj PRIVATE
                   "${root}/common/src/legacy_common_manager.cpp")
    target_compile_definitions(${OPHOST_NAME}_opapi_obj PRIVATE
                               NN_ENABLE_DLOPEN_LEGACY)
  endif()
endfunction()

function(batch_invariant_finalize root)
  # The transformer build snapshots the concrete op_host_aclnn* targets when
  # deciding which generated ACLNN sources to create. Imported operator CMake
  # files record their definitions on interface collectors instead, so mirror
  # only the BI definitions onto the concrete targets before that snapshot.
  foreach(kind aclnn aclnn_inner aclnn_exclude)
    if(kind STREQUAL "aclnn")
      set(concrete_target op_host_aclnn)
    elseif(kind STREQUAL "aclnn_inner")
      set(concrete_target op_host_aclnnInner)
    else()
      set(concrete_target op_host_aclnnExc)
    endif()
    set(collector_target ${OPHOST_NAME}_opdef_${kind}_obj)
    if(NOT TARGET ${collector_target} OR NOT TARGET ${concrete_target})
      continue()
    endif()
    get_target_property(definition_sources ${collector_target} INTERFACE_SOURCES)
    foreach(source ${definition_sources})
      if(source MATCHES "third_party/ops_batchinvariant")
        target_sources(${concrete_target} PRIVATE "${source}")
      endif()
    endforeach()
  endforeach()

  set(bi_quote_options
      "-iquote${root}"
      "-iquote${root}/ops/ascendc"
      "-iquote${root}/common/inc"
      "-iquote${root}/common/inc/op_api"
      "-iquote${root}/common/inc/op_host"
      "-iquote${root}/common/inc/op_graph"
      "-iquote${root}/common/inc/framework"
      "-iquote${ASCEND_CANN_PACKAGE_PATH}/${SYSTEM_PREFIX}/asc/include")

  foreach(target
          ${OPHOST_NAME}_opapi_obj
          ${OPHOST_NAME}_infer_obj
          ${OPHOST_NAME}_tiling_obj)
    if(NOT TARGET ${target})
      continue()
    endif()
    get_target_property(target_sources ${target} SOURCES)
    set(marked_sources 0)
    foreach(source ${target_sources})
      if(source MATCHES "third_party/ops_batchinvariant")
        set_property(SOURCE "${source}" TARGET_DIRECTORY ${target}
                     APPEND PROPERTY COMPILE_OPTIONS ${bi_quote_options})
        math(EXPR marked_sources "${marked_sources} + 1")
      endif()
    endforeach()
    message(STATUS "BI compatibility: ${target} has ${marked_sources} isolated sources")
  endforeach()
endfunction()
