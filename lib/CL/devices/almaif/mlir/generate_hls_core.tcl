# generate_hls_core.tcl - Vitis HLS script for converting Vitis Cpp kernel function
# 		          -> RTL accelerator
#
#   Copyright (c) 2025 Topi Leppänen / Tampere University
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
#   IN THE SOFTWARE.

set kernel_name [lindex $argv 2]
set input_file [lindex $argv 3]
set output_file [lindex $argv 4]
set vitis_folder_path [lindex $argv 5]

open_project ${vitis_folder_path}
add_files ${input_file}
set_top ${kernel_name}
open_solution "solution1" -flow_target vitis
set_part {xcu280-fsvh2892-2L-e}

create_clock -period 4 -name default
config_export -format ip_catalog -output hls_${kernel_name} -rtl verilog
config_export -ipname ${kernel_name}

set_directive_top -name ${kernel_name} "${kernel_name}"
csynth_design

export_design -rtl verilog -format ip_catalog -output ${output_file}

exit
