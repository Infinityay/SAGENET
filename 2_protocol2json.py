
import os
import subprocess
import json
import re
from pathlib import Path
import concurrent.futures
from tqdm import tqdm

# 配置要处理的协议列表
protocols = ['ntp'
             ]  # 只处理这些协议

# Define paths
input_dir = './data/protocol'
output_json_dir = './data/processed/json'
output_jsonraw_dir = './data/processed/jsonraw'

# Ensure output directories exist
Path(output_json_dir).mkdir(parents=True, exist_ok=True)
Path(output_jsonraw_dir).mkdir(parents=True, exist_ok=True)

# Get pcap files for specified protocols only
pcap_files = [f for f in os.listdir(input_dir) 
              if f.endswith('.pcap') and f.split('.')[0] in protocols]

print(f"找到 {len(pcap_files)} 个匹配的PCAP文件: {pcap_files}")

# 定义处理单个PCAP文件的函数
def process_pcap(pcap_file):
    protocol = pcap_file.split('.')[0]
    input_path = os.path.join(input_dir, pcap_file)
    
    # Output paths
    json_output = os.path.join(output_json_dir, f"{protocol}.json")
    jsonraw_output = os.path.join(output_jsonraw_dir, f"{protocol}_raw.json")
    
    # Command 1: Generate regular JSON
    json_command = ["tshark", "-r", input_path, "-T", "json"]
    with open(json_output, 'w', encoding='utf-8') as f:
        subprocess.run(json_command, stdout=f, check=True)
    
    # Command 2: Generate JSONraw with hex data
    jsonraw_command = ["tshark", "-r", input_path, "-T", "jsonraw", "-x"]
    
    # First save to a temporary file
    temp_jsonraw = f"{jsonraw_output}_temp"
    with open(temp_jsonraw, 'w', encoding='utf-8') as f:
        subprocess.run(jsonraw_command, stdout=f, check=True)
    
    # Read the temp file, process it, and write to the final file
    with open(temp_jsonraw, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
            
            # Write processed JSON to final file
            with open(jsonraw_output, 'w', encoding='utf-8') as out_f:
                # Convert to string with proper indentation
                json_str = json.dumps(data, ensure_ascii=False, indent=2)
                
                # Use regex to find array patterns and convert multi-line arrays to single line
                json_str = re.sub(r'\[\s+([^[]*?)\s+\]', 
                                lambda m: '[' + m.group(1).replace('\n', ' ').replace('  ', '') + ']', 
                                json_str)
                
                # Write to file
                out_f.write(json_str)
            
            # Remove temporary file
            os.remove(temp_jsonraw)
            
        except json.JSONDecodeError as e:
            print(f"错误处理 {temp_jsonraw}: {e}")
            # If error occurs, just use the original file
            os.rename(temp_jsonraw, jsonraw_output)
    
    return f"已处理 {pcap_file}"

# 使用线程池并行处理所有PCAP文件
print(f"开始并行处理 {len(pcap_files)} 个PCAP文件...")

# 设置最大工作线程数，可根据系统资源调整
max_workers = min(len(pcap_files), os.cpu_count() * 2)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # 创建任务并使用tqdm显示进度
    results = list(tqdm(
        executor.map(process_pcap, pcap_files), 
        total=len(pcap_files), 
        desc="转换PCAP文件"
    ))

print("所有PCAP文件已并行转换为JSON和JSONraw格式")
