# generate_xo.tcl - Vivado script for wrapping Almaif accelerator into Vivado block design
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

set project_path [lindex $argv 0]
set output_path [lindex $argv 1]
set kernel_count [lindex $argv 2]
set rtl_path [lindex $argv 3]

set idx 0
set kernel_argv_offset [expr $idx * 4 + 4]
set kernel_names [lindex $argv $kernel_argv_offset]
set wg_name_idx [expr $kernel_argv_offset + 1]
set workgroup_function_names [lindex $argv $wg_name_idx]
set vitis_path_idx [expr $kernel_argv_offset + 2]
set vitis_paths [lindex $argv $vitis_path_idx]
set ip_repo_idx [expr $kernel_argv_offset + 3]
set ip_repo_paths [lindex $argv $ip_repo_idx]
for {set idx 1} {$idx < ${kernel_count}} {incr idx} {
	set kernel_argv_offset [expr $idx * 4 + 4]
	lappend kernel_names [lindex $argv $kernel_argv_offset]
	set wg_name_idx [expr $kernel_argv_offset + 1]
	lappend workgroup_function_names [lindex $argv $wg_name_idx]
	set vitis_path_idx [expr $kernel_argv_offset + 2]
	lappend vitis_paths [lindex $argv $vitis_path_idx]
	set ip_repo_idx [expr $kernel_argv_offset + 3]
	lappend ip_repo_paths [lindex $argv $ip_repo_idx]
}
set ip_repo_path [lindex $ip_repo_paths 0]

create_project vivado_xo ${project_path} -part xcu280-fsvh2892-2L-e
set_property board_part xilinx.com:au280:part0:1.2 [current_project]

add_files [list $rtl_path/platform $rtl_path/gcu_ic $rtl_path/vhdl]

import_files -force
create_bd_design vec_kernel
update_compile_order -fileset sources_1
#set_property  ip_repo_paths [list ${ip_repo_path} ${vitis_path}] [current_project]
set_property ip_repo_paths ${vitis_paths} [current_project]
update_ip_catalog

######## Add the tta-core
create_bd_cell -type module -reference tta_core_toplevel tta_core_toplevel_0
set_property -dict [list CONFIG.local_mem_addrw_g {12} CONFIG.axi_addr_width_g {16} CONFIG.axi_offset_low_g {1073741824}] [get_bd_cells tta_core_toplevel_0]

####### Make tta-core s-axi external and connect rst and clk signals
make_bd_intf_pins_external  [get_bd_intf_pins tta_core_toplevel_0/s_axi]
set_property name s_axi_control [get_bd_intf_ports s_axi_0]
make_bd_pins_external  [get_bd_pins tta_core_toplevel_0/clk]
set_property name ap_clk [get_bd_ports clk_0]
make_bd_pins_external  [get_bd_pins tta_core_toplevel_0/rstx]
set_property name ap_rst_n [get_bd_ports rstx_0]

####### Create axi interconnect for connecting the tta-core to the DMAs
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_interconnect_0
set_property -dict [list CONFIG.NUM_SI {1} CONFIG.NUM_MI $kernel_count] [get_bd_cells axi_interconnect_0]
connect_bd_intf_net [get_bd_intf_pins tta_core_toplevel_0/m_axi] -boundary_type upper [get_bd_intf_pins axi_interconnect_0/S00_AXI]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_interconnect_0/ARESETN]
connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_interconnect_0/S00_ARESETN]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_interconnect_0/ACLK]
connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_interconnect_0/S00_ACLK]

######## Add the streaming functional unit
for {set idx 0} {$idx < ${kernel_count}} {incr idx} {
	set idx_padded [format "%02d" $idx]
	set workgroup_function_name [lindex $workgroup_function_names $idx]
	create_bd_cell -type ip -vlnv xilinx.com:hls:${workgroup_function_name}:1.0 ${workgroup_function_name}_ip_0
	connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins ${workgroup_function_name}_ip_0/ap_rst_n]
	connect_bd_net [get_bd_ports ap_clk] [get_bd_pins ${workgroup_function_name}_ip_0/ap_clk]
	make_bd_intf_pins_external  [get_bd_intf_pins ${workgroup_function_name}_ip_0/m_axi_*]
	connect_bd_intf_net -boundary_type upper [get_bd_intf_pins axi_interconnect_0/M${idx_padded}_AXI] [get_bd_intf_pins ${workgroup_function_name}_ip_0/s_axi_control]
	connect_bd_net [get_bd_ports ap_rst_n] [get_bd_pins axi_interconnect_0/M${idx_padded}_ARESETN]
	connect_bd_net [get_bd_ports ap_clk] [get_bd_pins axi_interconnect_0/M${idx_padded}_ACLK]
}

####### Create external axi masters
assign_bd_address
for {set idx 0} {$idx < ${kernel_count}} {incr idx} {
	set workgroup_function_name [lindex $workgroup_function_names $idx]
	set_property offset 0x0000000000 [get_bd_addr_segs ${workgroup_function_name}_ip_0/Data_m_axi_*/SEG_m_axi_*_Reg]
	set_property range 256M [get_bd_addr_segs ${workgroup_function_name}_ip_0/Data_m_axi_*/SEG_m_axi_*_Reg]
	# dummy set first, to get out of the valid range
	set control_offset [expr $idx * 0x10000 + ${kernel_count} * 0x10000]
	set_property offset ${control_offset} [get_bd_addr_segs tta_core_toplevel_0/m_axi/SEG_${workgroup_function_name}_ip_0_Reg]
}
# Now set the proper range, no risk of overlapping anymore
for {set idx 0} {$idx < ${kernel_count}} {incr idx} {
	set workgroup_function_name [lindex $workgroup_function_names $idx]
	set control_offset [expr $idx * 0x10000]
	set_property offset ${control_offset} [get_bd_addr_segs tta_core_toplevel_0/m_axi/SEG_${workgroup_function_name}_ip_0_Reg]
}
####### Set associated clock for the external interfaces
set list_m_axi [get_bd_intf_ports /m_axi_*]
foreach m_axi_tmp $list_m_axi {
	set m_axi [string range ${m_axi_tmp} 1 end]
	set_property CONFIG.ASSOCIATED_BUSIF ${m_axi} [get_bd_ports /ap_clk]
}

set_property CONFIG.ASSOCIATED_BUSIF s_axi_control [get_bd_ports /ap_clk]


####### Package project
regenerate_bd_layout
save_bd_design
ipx::package_project -root_dir ${ip_repo_path} -vendor user.org -library user -taxonomy /UserIP -module vec_kernel -import_files

####### Set secret registers according to the vitis doc
ipx::add_register CTRL [ipx::get_address_blocks Reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]]

set idx 0
foreach m_axi_tmp $list_m_axi {
	set m_axi [string range ${m_axi_tmp} 1 end]

	set dummy_offset [expr $idx * 0x8 + 0x10]
	ipx::add_register dummy${idx} [ipx::get_address_blocks Reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]]
	set_property address_offset ${dummy_offset} [ipx::get_registers dummy${idx} -of_objects [ipx::get_address_blocks Reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]]]
	set_property size 64 [ipx::get_registers dummy${idx} -of_objects [ipx::get_address_blocks Reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]]]
	ipx::add_register_parameter ASSOCIATED_BUSIF [ipx::get_registers dummy${idx} -of_objects [ipx::get_address_blocks Reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]]]
	set_property value ${m_axi} [ipx::get_register_parameters ASSOCIATED_BUSIF -of_objects [ipx::get_registers dummy${idx} -of_objects [ipx::get_address_blocks Reg0 -of_objects [ipx::get_memory_maps s_axi_control -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]]]]

	set idx [expr $idx + 1]
}


####### The vitis doc recommends this
ipx::remove_bus_parameter FREQ_HZ [ipx::get_bus_interfaces CLK.AP_CLK -of_objects [ipx::find_open_core user.org:user:vec_kernel:1.0]]

####### Set some parameters
set_property ipi_drc {ignore_freq_hz true} [ipx::find_open_core user.org:user:vec_kernel:1.0]
set_property sdx_kernel true [ipx::find_open_core user.org:user:vec_kernel:1.0]
set_property sdx_kernel_type rtl [ipx::find_open_core user.org:user:vec_kernel:1.0]
set_property vitis_drc {ctrl_protocol ap_ctrl_hs} [ipx::find_open_core user.org:user:vec_kernel:1.0]
set_property vitis_drc {ctrl_protocol user_managed} [ipx::find_open_core user.org:user:vec_kernel:1.0]
set_property ipi_drc {ignore_freq_hz true} [ipx::find_open_core user.org:user:vec_kernel:1.0]

####### ?
ipx::infer_bus_interface ap_rst_n xilinx.com:signal:reset_rtl:1.0 [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::associate_bus_interfaces -clock CLK.AP_CLK -reset ap_rst_n [ipx::find_open_core user.org:user:vec_kernel:1.0]

#######??
ipx::update_checksums [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::save_core [ipx::find_open_core user.org:user:vec_kernel:1.0]
set_property core_revision 2 [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::create_xgui_files [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::update_checksums [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::check_integrity -kernel [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::save_core [ipx::find_open_core user.org:user:vec_kernel:1.0]

ipx::check_integrity -quiet -kernel [ipx::find_open_core user.org:user:vec_kernel:1.0]
ipx::archive_core ${ip_repo_path}/user.org_user_vec_kernel_1.0.zip [ipx::find_open_core user.org:user:vec_kernel:1.0]

package_xo  -xo_path ${output_path} -kernel_name pocl_kernel -ip_directory ${ip_repo_path} -ctrl_protocol user_managed

