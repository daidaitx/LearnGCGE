# 定义 PETSc 作为 INTERFACE 库
add_library(PETSC INTERFACE)

# 判断当前系统
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(PETSC_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/windows)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(PETSC_DIR ${PROJECT_SOURCE_DIR}/3rd/petsc/linux)
else()
    message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

set(PETSC_INCLUDE_DIR ${PETSC_DIR}/include)
set(PETSC_LIB_DIR ${PETSC_DIR}/lib)
set(PETSC_LIB_NAME petsc)

# 设置头文件路径
target_include_directories(PETSC INTERFACE ${PETSC_INCLUDE_DIR})

# 设置库文件搜索路径
target_link_directories(PETSC INTERFACE ${PETSC_LIB_DIR})

# 链接 PETSc 库
target_link_libraries(PETSC INTERFACE ${PETSC_LIB_NAME})
