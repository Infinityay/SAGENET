"""
从大型PCAP文件中按协议提取指定数量的数据包 - 优化版

用法:
    直接在代码中设置参数并运行
"""

import subprocess
import sys
import os
import glob
from pathlib import Path
import tempfile
import shutil
import concurrent.futures
from tqdm import tqdm  # 添加进度条支持
import time

# 协议映射表，将用户输入的协议名称映射到tshark过滤表达式
PROTOCOL_FILTERS = {
    # 常见网络协议
    "arp": "arp.opcode == 1",  # 只获取ARP request包
    "icmp": "(icmp.type == 8 and icmp.code == 0 and !icmp.data_time) && (data.len == 56)", # 只获取icmp echo request包 且不要data_time数据
    "tcp": "tcp",
    "udp": "udp && frame.len<110",
    "dns": "dns.qry.name.len == 15 and dns.flags.response == 0", # 只获取dns query包 
    "nbns": "nbns.flags.opcode == 0", # 只获取nbns Name query包
    # "ntp": "ntp.flags.vn == 4 and ntp.flags.mode==3", # 只获取ntp version 4的client包
    "ntp": "ntp.flags.vn == 4 and ntp.flags.mode==4",     # ntp server
    "smb": "smb.flags.response == 0", 
    # 工业协议
    "modbus": '(((modbus and !modbus.request_frame)) and (modbus.func_code == 4)) && (modbus.reference_num != 0)', # 只获取Query包 
    "s7comm": "s7comm.header.rosctr == 3", # 只获取ACK_DATA 包
}



def get_protocol_filter(protocol_name):
    """获取协议对应的tshark过滤表达式"""
    # 转换为小写并去除空格
    protocol_name = protocol_name.lower().strip()
    
    # 如果直接在映射表中，则直接返回
    if protocol_name in PROTOCOL_FILTERS:
        return PROTOCOL_FILTERS[protocol_name]
    
    # 否则尝试作为直接过滤表达式返回
    return protocol_name

def check_file_exists_and_has_size(file_path, min_size=1):
    """检查文件是否存在且大小大于最小值"""
    path = Path(file_path)
    return path.exists() and path.stat().st_size >= min_size

def count_packets(pcap_file):
    """统计PCAP文件中的数据包数量"""
    try:
        # 使用capinfos统计数据包数量，这更可靠
        command = ["capinfos", "-c", str(pcap_file)]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 从capinfos输出中提取数据包数量
        packet_count = "未知"
        for line in process.stdout.splitlines():
            if "Number of packets" in line:
                packet_count = line.split(":")[-1].strip()
                return packet_count
        
        # 如果capinfos命令成功但没有找到数量信息，尝试使用tshark
        command = ["tshark", "-r", str(pcap_file), "-c", "1"]
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )
        return "至少1个"
    except Exception as e:
        # 返回字符串"未知"而不是抛出异常
        return "未知"

def merge_pcap_files(input_files, output_file, verbose=False):
    """
    合并多个PCAP文件为一个文件
    
    参数:
        input_files: 输入PCAP文件路径列表
        output_file: 输出PCAP文件路径
        verbose: 是否显示详细信息
    
    返回:
        bool: 合并是否成功
    """
    if len(input_files) == 0:
        print("错误: 没有指定输入文件", file=sys.stderr)
        return False
    
    if len(input_files) == 1:
        print(f"只有一个输入文件，复制 {input_files[0]} 到 {output_file}")
        shutil.copy2(input_files[0], output_file)
        return True
    
    # 检查所有输入文件是否存在
    for input_file in input_files:
        if not Path(input_file).exists():
            print(f"错误: 输入文件 '{input_file}' 不存在", file=sys.stderr)
            return False
    
    # 构建mergecap命令
    command = ["mergecap", "-w", str(output_file)]
    command.extend([str(f) for f in input_files])
    
    if verbose:
        print(f"执行合并命令: {' '.join(command)}")
    
    try:
        # 使用tqdm显示进度
        print("正在合并文件...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # 简单进度显示
        with tqdm(total=100, desc="合并进度") as pbar:
            while process.poll() is None:
                pbar.update(1)
                time.sleep(0.1)
                if pbar.n >= 99:
                    pbar.n = 90  # 重置进度，直到进程完成
            pbar.n = 100
            pbar.refresh()
        
        # 检查返回码
        if process.returncode != 0:
            stderr = process.stderr.read()
            print(f"错误: 执行mergecap命令时出错: {stderr}", file=sys.stderr)
            return False
        
        if check_file_exists_and_has_size(output_file):
            packet_count = count_packets(output_file)
            file_size = Path(output_file).stat().st_size // (1024 * 1024)  # 转换为MB
            print(f"成功: 已合并 {len(input_files)} 个PCAP文件到 {output_file}")
            print(f"合并文件大小: {file_size}MB, 数据包数量: {packet_count}")
            return True
        else:
            print(f"错误: 合并后的文件 {output_file} 不存在或为空", file=sys.stderr)
            return False
    
    except Exception as e:
        print(f"合并文件时发生未知错误: {e}", file=sys.stderr)
        return False

def get_total_packets_for_protocol(input_pcap, protocol, verbose=False):
    """获取特定协议的总数据包数量"""
    filter_expr = get_protocol_filter(protocol)
    
    try:
        # 使用tshark读取所有匹配数据包并计数，不保存输出
        command = ["tshark", "-r", str(input_pcap), "-Y", filter_expr]
        
        if verbose:
            print(f"执行计数命令: {' '.join(command)}")
        
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False  # 不检查返回值，避免大文件导致异常
        )
        
        # 计算输出行数
        lines = process.stdout.strip().split('\n')
        packet_count = len([line for line in lines if line.strip()])
        
        return packet_count
    except Exception as e:
        print(f"警告: 无法统计协议 {protocol} 的数据包数量: {e}", file=sys.stderr)
        return 0

def extract_protocol_by_chunks(input_pcap, protocol, output_dir, count=None, verbose=False):
    """
    通过分块方式提取特定协议的数据包
    
    这种方法更适合处理大型pcap文件，可以确保每个协议都提取到指定数量的数据包
    """
    filter_expr = get_protocol_filter(protocol)
    output_path = output_dir / f"{protocol.lower().replace('/', '_').replace(' ', '_')}.pcap"
    
    if verbose:
        print(f"\n处理协议: {protocol}")
        print(f"过滤表达式: {filter_expr}")
        print(f"输出文件: {output_path}")
        if count:
            print(f"目标数据包数量: {count}")
    
    # 先检查协议总共有多少个数据包
    total_packets = get_total_packets_for_protocol(input_pcap, protocol, verbose)
    if verbose:
        print(f"源文件中共找到 {total_packets} 个 {protocol} 数据包")
    
    if total_packets == 0:
        print(f"警告: 在源文件中没有找到任何 {protocol} 数据包")
        return False
    
    # 确定实际提取的数据包数量
    actual_count = count if count and count < total_packets else total_packets
    if verbose:
        print(f"将提取 {actual_count} 个 {protocol} 数据包")
    
    # 创建临时目录来存储中间pcap文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / f"{protocol}_temp.pcap"
        
        # 提取指定协议的所有数据包到临时文件
        extract_cmd = ["tshark", "-r", str(input_pcap), "-Y", filter_expr, "-w", str(temp_path)]
        
        if verbose:
            print(f"执行提取命令: {' '.join(extract_cmd)}")
        
        try:
            subprocess.run(extract_cmd, capture_output=True, text=True, check=True)
            
            if not check_file_exists_and_has_size(temp_path):
                print(f"错误: 未能提取任何 {protocol} 数据包到临时文件")
                return False
            
            # 如果要限制数量，从临时文件中提取指定数量的数据包
            if count and count < total_packets:
                trim_cmd = ["editcap", "-r", str(temp_path), str(output_path), "1-{}".format(count)]
                
                if verbose:
                    print(f"执行截取命令: {' '.join(trim_cmd)}")
                
                subprocess.run(trim_cmd, capture_output=True, text=True, check=True)
            else:
                # 否则直接复制临时文件到输出路径
                shutil.copy2(temp_path, output_path)
            
            # 验证输出文件是否存在并有内容
            if check_file_exists_and_has_size(output_path):
                packet_count = count_packets(output_path)
                print(f"成功: 已提取 {packet_count} 个 {protocol} 数据包到 {output_path}")
                return True
            else:
                print(f"错误: 输出文件不存在或为空")
                return False
            
        except subprocess.CalledProcessError as e:
            print(f"错误: 处理协议 {protocol} 时出错: {e}", file=sys.stderr)
            if verbose and e.stderr:
                print(f"标准错误输出:\n{e.stderr}", file=sys.stderr)
            
            # 检查是否是协议识别错误
            if e.stderr and "neither a field nor a protocol" in e.stderr:
                print(f"提示: 过滤表达式 '{filter_expr}' 不是有效的协议或字段名称。")
                # 尝试查找相似的协议名称作为建议
                suggestions = [p for p in PROTOCOL_FILTERS.keys() if p.startswith(protocol[:2]) or protocol[:2] in p]
                if suggestions:
                    print(f"您是否想要使用以下协议之一? {', '.join(suggestions)}")
            
            return False
        except Exception as e:
            print(f"处理协议 {protocol} 时发生未知错误: {e}", file=sys.stderr)
            return False

def batch_extract_protocols(input_pcap, protocols, output_dir, count=None, verbose=False, max_workers=None):
    """
    批量并行提取多个协议的数据包
    
    参数:
        input_pcap: 输入PCAP文件路径
        protocols: 要提取的协议列表
        output_dir: 输出目录路径
        count: 每个协议提取的数据包数量限制
        verbose: 是否显示详细信息
        max_workers: 最大工作线程数，默认为CPU核心数的2倍
    
    返回:
        tuple: (成功提取的协议数量, 失败的协议列表)
    """
    # 确定最大工作线程数
    if max_workers is None:
        max_workers = min(len(protocols), os.cpu_count() * 2)
    
    successful_extractions = 0
    failed_protocols = []
    
    # 使用线程池并行处理
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 创建任务
        future_to_protocol = {
            executor.submit(extract_protocol_by_chunks, input_pcap, protocol, output_dir, count, verbose): protocol
            for protocol in protocols
        }
        
        # 使用tqdm添加进度条
        for future in tqdm(concurrent.futures.as_completed(future_to_protocol), total=len(future_to_protocol), desc="提取协议"):
            protocol = future_to_protocol[future]
            try:
                success = future.result()
                if success:
                    successful_extractions += 1
                else:
                    failed_protocols.append(protocol)
            except Exception as e:
                print(f"处理协议 {protocol} 时发生错误: {e}")
                failed_protocols.append(protocol)
    
    return successful_extractions, failed_protocols

def process_pcap_files(protocols, input_dir, output_dir, count=None, verbose=False, merge_only=False, max_workers=None):
    """
    处理PCAP文件的主函数 - 优化版
    
    参数:
        protocols: 要提取的协议列表
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        count: 每个协议提取的数据包数量限制
        verbose: 是否显示详细信息
        merge_only: 是否只合并PCAP文件，不提取协议
        max_workers: 最大工作线程数，默认为CPU核心数的2倍
    """
    start_time = time.time()
    
    # 固定设定要处理的 pcap 文件路径
    input_files = [
        Path("data/input/16-11-17.pcap"),
        Path("data/input/16-09-27.pcap"),
        # Path("./data/input/s7comm_yh.pcap")

    ]

    # 检查文件是否存在
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        print(f"错误: 以下文件不存在: {[str(f) for f in missing_files]}", file=sys.stderr)
        return

    print(f"将处理以下固定的pcap文件: {[f.name for f in input_files]}")
    
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置合并后的文件路径
    merged_file = output_dir / "merged.pcap"
    
    # 输出基本信息
    print(f"输入文件: {', '.join(str(f) for f in input_files)}")
    print(f"输出目录: {output_dir}")
    if not merge_only:
        print(f"要提取的协议: {', '.join(protocols)}")
        if count:
            print(f"每个协议提取的数据包数量限制: {count}")
    
    # 检查工具是否可用
    tools = ["tshark", "editcap", "capinfos", "mergecap"]
    missing_tools = []
    
    for tool in tools:
        try:
            subprocess.run([tool, "--version"], capture_output=True, check=True)
        except FileNotFoundError:
            missing_tools.append(tool)
        except subprocess.CalledProcessError:
            print(f"警告: {tool} 命令可用，但返回了错误。继续处理，但可能会遇到问题。", file=sys.stderr)
    
    if missing_tools:
        print(f"错误: 缺少以下工具: {', '.join(missing_tools)}。请确保已安装 Wireshark 并将其添加到系统 PATH。", file=sys.stderr)
        return
    
    # 合并PCAP文件
    print("\n合并PCAP文件...")
    if not merge_pcap_files(input_files, merged_file, verbose):
        print("错误: 合并PCAP文件失败", file=sys.stderr)
        return
    
    # 如果只需要合并，到此结束
    if merge_only:
        print(f"\n合并完成。输出文件: {merged_file}")
        end_time = time.time()
        print(f"总耗时: {end_time - start_time:.2f} 秒")
        return
    
    # 并行处理每个协议
    print("\n开始并行提取协议...")
    successful_extractions, failed_protocols = batch_extract_protocols(
        merged_file, protocols, output_dir, count, verbose, max_workers
    )
    
    # 输出汇总信息
    print(f"\n处理完成。")
    print(f"成功提取的协议: {successful_extractions}/{len(protocols)}")
    if failed_protocols:
        print(f"失败的协议: {', '.join(failed_protocols)}")
    print(f"输出目录: {output_dir}")
    print(f"合并后的文件: {merged_file}")
    
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    # 定义参数
    protocols = ['ntp']
    
    input_dir = './data/input/'
    output_dir = './data/protocol/'
    count = 2000
    verbose = False  # 是否显示详细输出
    merge_only = False  # 是否只合并PCAP文件，不提取协议
    max_workers = None  # 最大工作线程数，默认为CPU核心数的2倍
    
    # 处理PCAP文件
    process_pcap_files(protocols, input_dir, output_dir, count, verbose, merge_only, max_workers)

    # 删除合并后的文件
    merged_file = Path(output_dir) / "merged.pcap"
    if merged_file.exists() and not merge_only:
        print(f"清理: 删除临时合并文件 {merged_file}")
        try:
            merged_file.unlink()
        except Exception as e:
            print(f"警告: 删除合并文件时出错: {e}", file=sys.stderr)
