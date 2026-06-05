find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Find nanobind (installed via pip in build-system.requires)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

# Use nanobind_add_module for proper scikit-build-core integration.
# FREE_THREADED marks the module Py_MOD_GIL_NOT_USED so it loads without
# re-enabling the GIL on free-threaded CPython builds (3.13t, 3.14t, ...).
nanobind_add_module(_ale_py MODULE FREE_THREADED ale_python_interface.cpp)
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_XLA_LIB)
    # XLA Integration setup
    message(STATUS "Python Executable: '${Python_EXECUTABLE}'")
    message(STATUS "Python_SITELIB=${Python_SITELIB}, Python_SITEARCH=${Python_SITEARCH}")
    execute_process(
        COMMAND "${Python_EXECUTABLE}" "-c"
        "import sys; sys.path.append(r'${Python_SITELIB}'); sys.path.append(r'${Python_SITEARCH}'); from jax import ffi; print(ffi.include_dir())"
        OUTPUT_STRIP_TRAILING_WHITESPACE
        OUTPUT_VARIABLE XLA_DIR
    )
    message(STATUS "XLA include directory: '${XLA_DIR}'")

    # Find CUDA toolkit for headers and linking (optional - XLA disabled if not found)
    find_package(CUDAToolkit QUIET)

    if (CUDAToolkit_FOUND)
        message(STATUS "CUDA Toolkit found: ${CUDAToolkit_VERSION}")
        target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp ale_vector_xla_interface.cpp)
        target_include_directories(_ale_py PUBLIC ${XLA_DIR})
        target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB BUILD_VECTOR_XLA_LIB)
        target_link_libraries(_ale_py PRIVATE ${CMAKE_DL_LIBS})
    else()
        message(WARNING "CUDA Toolkit not found - disabling XLA support. Install CUDA Toolkit to enable XLA.")
        target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp)
        target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB)
    endif()
elseif (BUILD_VECTOR_LIB)
    target_sources(_ale_py PRIVATE ale_vector_python_interface.cpp)
    target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_LIB)
endif()

# PyTorch C++ backend for the vector env (opt-in: -DBUILD_VECTOR_TORCH_LIB=ON).
# Compiled into _ale_py so it shares the single EnvVectorizer instance; the
# libtorch dependency therefore only exists in these opt-in builds, not in the
# standard wheels. We query libtorch's headers/libs from the Python install
# rather than find_package(Torch), whose Caffe2 CUDA cmake rejects a system CUDA
# toolkit that mismatches torch's bundled one (we don't compile CUDA kernels:
# cudart is loaded via dlsym and only c10's stream API is linked).
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_TORCH_LIB)
    execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
        "import torch.utils.cpp_extension as c; print(';'.join(c.include_paths()))"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_INCLUDE_DIRS)
    execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
        "import os, torch; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_LIB_DIR)
    execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
        "import torch; print(1 if torch.version.cuda else 0)"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_HAS_CUDA)
    execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
        "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_ABI)

    target_sources(_ale_py PRIVATE ale_vector_torch_interface.cpp)
    target_compile_definitions(_ale_py PRIVATE
        BUILD_VECTOR_TORCH_LIB _GLIBCXX_USE_CXX11_ABI=${TORCH_ABI})
    target_include_directories(_ale_py PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_directories(_ale_py PRIVATE ${TORCH_LIB_DIR})
    target_link_libraries(_ale_py PRIVATE torch torch_cpu c10)

    # Enable the CUDA send/recv path when torch was built against CUDA.
    if (TORCH_HAS_CUDA STREQUAL "1")
        message(STATUS "PyTorch CUDA backend enabled for vector env")
        target_compile_definitions(_ale_py PRIVATE BUILD_VECTOR_TORCH_CUDA)
        target_link_libraries(_ale_py PRIVATE c10_cuda torch_cuda)
        # CUDA headers for <cuda_runtime_api.h> (from the nvidia pip wheel that
        # matches torch's CUDA version; fall back to a system CUDA toolkit).
        execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
            "import os, nvidia.cuda_runtime as m; print(os.path.join(os.path.dirname(m.__file__), 'include'))"
            OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NVIDIA_CUDA_INCLUDE
            ERROR_QUIET)
        if (NVIDIA_CUDA_INCLUDE AND IS_DIRECTORY "${NVIDIA_CUDA_INCLUDE}")
            target_include_directories(_ale_py PRIVATE ${NVIDIA_CUDA_INCLUDE})
        else()
            find_package(CUDAToolkit QUIET)
            if (CUDAToolkit_FOUND)
                target_include_directories(_ale_py PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
            endif()
        endif()
    endif()
endif()

target_link_libraries(_ale_py PUBLIC ale ale-lib)

# Install the compiled module
install(TARGETS _ale_py LIBRARY DESTINATION ale_py)

# Install Python source files
install(FILES
    __init__.py
    env.py
    registration.py
    vector_env.py
    _xla_ops.py
    _torch_ops.py
    _torchrl_env.py
    py.typed
    DESTINATION ale_py)

# Install stub files if they exist
install(FILES
    __init__.pyi
    DESTINATION ale_py
    OPTIONAL)

# Install roms directory
install(DIRECTORY roms/
    DESTINATION ale_py/roms
    FILES_MATCHING PATTERN "*")

# If we're dynamically loading SDL with Python we'll be building a wheel
# so we should prepare SDL for distribution. Add rpath and copy over
# the dynamic library. auditwheel will take care of ensuring
# cross-platform compatibility on macOS and Linux.
if (SDL_SUPPORT AND SDL_DYNLOAD)
    set_target_properties(_ale_py PROPERTIES
        INSTALL_RPATH_USE_ORIGIN TRUE
        BUILD_WITH_INSTALL_RPATH TRUE
        SKIP_BUILD_RPATH FALSE
        INSTALL_RPATH_USE_LINK_PATH FALSE
        MACOSX_RPATH TRUE
        INSTALL_RPATH
        "$<$<PLATFORM_ID:Darwin>:@loader_path>$<$<PLATFORM_ID:Linux>:\$ORIGIN>")

    # Define our SDL2 distribution library name for dynamic loading
    target_compile_definitions(ale
        PRIVATE SDL2_LIBRARY_NAME="$<TARGET_FILE_NAME:SDL2::SDL2>")
    # Copy over SDL2 dist. library
    add_custom_command(TARGET _ale_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_FILE:SDL2::SDL2>
            $<TARGET_FILE_DIR:_ale_py>)
    # Install SDL2 library alongside the module
    install(FILES $<TARGET_FILE:SDL2::SDL2> DESTINATION ale_py)
endif()

# Resolve libtorch at load time so `import ale_py` works without `import torch`
# first. Appended last so it survives the SDL block's INSTALL_RPATH assignment.
# ale_py/ -> ../torch/lib in site-packages; absolute build dir for in-tree runs.
if (BUILD_VECTOR_LIB AND BUILD_VECTOR_TORCH_LIB)
    set_property(TARGET _ale_py APPEND PROPERTY INSTALL_RPATH "$ORIGIN/../torch/lib")
    set_property(TARGET _ale_py APPEND PROPERTY BUILD_RPATH "${TORCH_LIB_DIR}")
endif()
