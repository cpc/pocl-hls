#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2019 Pekka Jääskeläinen / Tampere University
#                 2022 Topi Leppänen / Tampere University
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================


set(ALMAIF_SOURCES "AlmaifShared.hh"
                  "almaif.cc"
                  "almaif.h"
                  "AlmaIFRegion.cc"
                  "AlmaIFRegion.hh"
                  "AlmaIFDevice.cc"
                  "AlmaIFDevice.hh"
                  "MMAPRegion.cc"
                  "MMAPRegion.hh"
                  "MMAPDevice.cc"
                  "MMAPDevice.hh"
                  "EmulationRegion.cc"
                  "EmulationRegion.hh"
                  "EmulationDevice.cc"
                  "EmulationDevice.hh"
                  "XilinxXrtDevice.hh"
                  "altera/AlteraOpaeDevice.hh"
                  "AlmaifCompile.cc"
                  "AlmaifCompile.hh"
                  "openasip/AlmaifCompileOpenasip.hh"
                  "mlir/AlmaifCompileMLIR.hh"
                  "mlir/AlmaifCompileMLIR.cc"
                  )

if(HAVE_XRT)
    add_compile_options(-I${XRT_INCLUDEDIR})
    set(ALMAIF_SOURCES ${ALMAIF_SOURCES}
                      "XilinxXrtDevice.cc"
                      "XilinxXrtRegion.cc"
                      "XilinxXrtRegion.hh"
                      "XilinxXrtExternalRegion.cc"
                      "XilinxXrtExternalRegion.hh"
                      "AlmaifDB/DBDevice.hh"
                      "AlmaifDB/DBDevice.cc"
                      "AlmaifDB/AlmaIFBitstreamDatabaseManager.cc"
                      "AlmaifDB/AlmaIFBitstreamDatabaseManager.hh"
                      "AlmaifDB/tiny-json.c"
                      "AlmaifDB/tiny-json.h"
                      )
endif()

if(HAVE_OPAE)
    set(ALMAIF_SOURCES ${ALMAIF_SOURCES}
                    "altera/AlteraOpaeDevice.cc"
                    "altera/AlteraOpaeRegion.cc"
                    "altera/AlteraOpaeRegion.hh"
                    "altera/AlteraOpaeExternalRegion.cc"
                    "altera/AlteraOpaeExternalRegion.hh"
                    )
endif()

if(ENABLE_TCE)
    set(ALMAIF_SOURCES ${ALMAIF_SOURCES}
                      "openasip/TTASimDevice.hh"
                      "openasip/TTASimDevice.cc"
                      "openasip/TTASimRegion.cc"
                      "openasip/TTASimRegion.hh"
                      "openasip/TTASimControlRegion.cc"
                      "openasip/TTASimControlRegion.hh"
                      "openasip/AlmaifCompileOpenasip.cc"
                      )
endif()

add_pocl_device_library(pocl-devices-almaif ${ALMAIF_SOURCES})

if(HAVE_XRT)
    target_link_libraries(pocl-devices-almaif PRIVATE "${XRT_LIBDIR}/libxrt_coreutil.so")
endif()

if(HAVE_OPAE)
	target_include_directories(pocl-devices-almaif PRIVATE "${OPAE_BOARD_DIR}/source/host")
	target_include_directories(pocl-devices-almaif PRIVATE "${OPAE_BOARD_DIR}/linux64/include")
       	target_include_directories(pocl-devices-almaif PRIVATE "${OPAE_BOARD_DIR}/source/include")
       	target_include_directories(pocl-devices-almaif PRIVATE "${OPAE_BOARD_DIR}/source/extra/intel-fpga-bbb/BBB_cci_mpf/sw/include/")
	#add_compile_options("-Wl,--no-as-needed")
    	#add_compile_options("-L${OPAE_BOARD_DIR}/linux64/lib -L${ONEAPI_DIR}/host/linux64/lib")
    #target_link_libraries(pocl-device-almaif PRIVATE "-lopae-c -lintel_opae_mmd -lMPF -luuid -lelf -lstdc++")
    target_link_libraries(pocl-devices-almaif PRIVATE "opae-c")
    target_link_libraries(pocl-devices-almaif PRIVATE "uuid")
    target_link_libraries(pocl-devices-almaif PRIVATE "${OPAE_BOARD_DIR}/linux64/lib/libintel_opae_mmd.so")
    target_link_libraries(pocl-devices-almaif PRIVATE "${OPAE_BOARD_DIR}/linux64/lib/libMPF.so")
    target_link_libraries(pocl-devices-almaif PRIVATE "${ONEAPI_DIR}/host/linux64/lib/libelf.so")
endif()

if(ENABLE_TCE)
    target_link_libraries(pocl-devices-almaif PRIVATE ${TCE_LIBS})

    target_compile_options(pocl-devices-almaif PRIVATE ${TCE_INCLUDES})
    target_compile_options(pocl-devices-almaif PRIVATE ${TCE_CXXFLAGS})
endif()

install(FILES "tce_builtins.cl"
        DESTINATION "${POCL_INSTALL_PRIVATE_DATADIR_REL}/almaif"
	COMPONENT "lib")

set(ADF "${CMAKE_CURRENT_SOURCE_DIR}/mlir/tta.adf")
add_custom_command(
  OUTPUT rtl_tta_core/platform/tta_core_toplevel.vhdl
  COMMAND rm -rf rtl_tta_core
  COMMAND LD_LIBRARY_PATH=${OPENASIP_LLVM_DIR}/lib generateprocessor -t --prefer-generation
    -o rtl_tta_core -e tta_core --icd-arg-list=debugger:minimal
    --hdb-list=generate_lsu_32.hdb,generate_rf_iu.hdb,xilinx_series7.hdb,generate_base32.hdb
    -f onchip -d onchip -g AlmaIFIntegrator -p dummy.tpef ${ADF}
  COMMAND LD_LIBRARY_PATH=${OPENASIP_LLVM_DIR}/lib generatebits -x rtl_tta_core -e tta_core ${ADF}
  DEPENDS ${ADF})

add_custom_target(rtl_tta_core ALL
  DEPENDS rtl_tta_core/platform/tta_core_toplevel.vhdl)
