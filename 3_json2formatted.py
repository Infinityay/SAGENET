import json
import re
import subprocess
import os
from pathlib import Path
from typing import Dict, List, Optional, Generator, Any, Union

class ProtocolJsonHandler:
    """
    处理已预处理的协议 JSON 数据
    适用于已有 json/ 和 jsonraw/ 文件夹的情况
    """
    
    def __init__(self, json_folder: Path, jsonraw_folder: Path, include_protocols: List[str] = None):
        """
        初始化处理器
        
        Args:
            json_folder: 包含协议 JSON 文件的文件夹路径
            jsonraw_folder: 包含协议原始 JSON 文件的文件夹路径  
            include_protocols: 要包含的协议列表，None 表示包含所有协议
        """
        self.json_folder = Path(json_folder)
        self.jsonraw_folder = Path(jsonraw_folder)
        self.include_protocols = [p.lower() for p in include_protocols] if include_protocols else None
        
        # 获取可用的协议文件
        self.available_protocols = self._get_available_protocols()
        
        # 用于跟踪每个协议实际使用的包ID
        self.used_packet_ids = {}
    
    def _get_available_protocols(self) -> List[str]:
        """获取可用的协议列表"""
        protocols = []
        for json_file in self.json_folder.glob("*.json"):
            protocol_name = json_file.stem
            # 检查对应的 raw 文件是否存在
            raw_file = self.jsonraw_folder / f"{protocol_name}_raw.json"
            if raw_file.exists():
                protocols.append(protocol_name)
                
                # 如果是modbus协议，同时添加mbtcp作为可用协议
                if protocol_name.lower() == 'modbus':
                    protocols.append('mbtcp')
        
        return protocols
    
    def _validate_split_positions(self, split_pos: Dict, data_len: int) -> Dict:
        """
        验证分割位置的有效性
        
        Args:
            split_pos: 包含 b_pos 和 e_pos 的字典
            data_len: 数据长度（字节）
        
        Returns:
            验证结果字典
        """
        b_pos = split_pos['b_pos']
        e_pos = split_pos['e_pos']
        max_bits = data_len * 8
        
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # 1. 基本数量检查
        validation['stats']['b_pos_count'] = len(b_pos)
        validation['stats']['e_pos_count'] = len(e_pos)
        
        if len(b_pos) != len(e_pos):
            validation['errors'].append(f"Begin positions ({len(b_pos)}) != End positions ({len(e_pos)})")
            validation['is_valid'] = False
        
        # 2. 位置有效性检查
        for i, pos in enumerate(b_pos):
            if pos < 0:
                validation['errors'].append(f"Begin position {i} is negative: {pos}")
                validation['is_valid'] = False
            elif pos >= max_bits:
                validation['errors'].append(f"Begin position {i} exceeds data length: {pos} >= {max_bits}")
                validation['is_valid'] = False
        
        for i, pos in enumerate(e_pos):
            if pos < 0:
                validation['errors'].append(f"End position {i} is negative: {pos}")
                validation['is_valid'] = False
            elif pos > max_bits:
                validation['errors'].append(f"End position {i} exceeds data length: {pos} > {max_bits}")
                validation['is_valid'] = False
        
        # 3. 如果基本检查通过，进行逻辑检查
        if validation['is_valid'] and len(b_pos) == len(e_pos):
            # 检查每个字段的开始-结束位置对应关系
            fields = []
            for i in range(len(b_pos)):
                start = b_pos[i]
                # 找到对应的结束位置（最小的大于start的end位置）
                end = None
                for e in e_pos:
                    if e > start:
                        end = e
                        break
                
                if end is None:
                    validation['warnings'].append(f"Begin position {start} has no matching end position")
                else:
                    fields.append((start, end))
            
            # 按开始位置排序
            fields.sort()
            validation['stats']['valid_fields'] = len(fields)
            
            # 4. 检查字段重叠和间隙
            for i in range(len(fields) - 1):
                current_end = fields[i][1]
                next_start = fields[i + 1][0]
                
                if current_end > next_start:
                    validation['warnings'].append(
                        f"Field overlap: field {i} ends at {current_end}, field {i+1} starts at {next_start}"
                    )
                elif current_end < next_start:
                    gap_size = next_start - current_end
                    if gap_size > 8:  # 超过1字节的间隙可能有问题
                        validation['warnings'].append(
                            f"Large gap ({gap_size} bits) between field {i} and {i+1}"
                        )
            
            # 5. 检查空字段
            empty_fields = [(start, end) for start, end in fields if start == end]
            if empty_fields:
                validation['warnings'].append(f"Found {len(empty_fields)} empty fields: {empty_fields}")
            
            # 6. 覆盖率统计
            if fields:
                total_covered_bits = sum(end - start for start, end in fields if end > start)
                coverage_percentage = (total_covered_bits / max_bits) * 100
                validation['stats']['coverage_percentage'] = round(coverage_percentage, 2)
                validation['stats']['total_covered_bits'] = total_covered_bits
                validation['stats']['max_bits'] = max_bits
                
                if coverage_percentage < 50:
                    validation['warnings'].append(f"Low field coverage: {coverage_percentage:.1f}%")
                elif coverage_percentage > 100:
                    validation['errors'].append(f"Field coverage exceeds 100%: {coverage_percentage:.1f}%")
                    validation['is_valid'] = False
        
        # 7. 字节对齐检查
        unaligned_positions = [pos for pos in b_pos + e_pos if pos % 8 != 0]
        if unaligned_positions:
            validation['stats']['unaligned_positions'] = len(unaligned_positions)
            # 这通常是正常的（位字段），所以只是统计而不是警告
        
        return validation
    
    @staticmethod
    def _select_first(*items, fun=lambda x: x is not None):
        """从多个选项中选择第一个满足条件的"""
        for item in items:
            if fun(item):
                return item
        return None
    
    @staticmethod
    def _get_value(obj: Dict, keys: List[str], default=None):
        """安全地从嵌套字典中获取值"""
        try:
            result = obj
            for key in keys:
                result = result[key]
            return result
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def _parse_bitmask(field_info: List) -> tuple:
        """解析位掩码信息"""
        if len(field_info) < 4:
            return -1, -1
        
        mask_bin = f'{field_info[3]:0>{field_info[2] * 8}b}'
        l_pos = mask_bin.find('1')
        r_pos = mask_bin.rfind('1')
        
        if l_pos >= 0 and r_pos >= 0:
            return l_pos, r_pos + 1
        else:
            return -1, -1
    
    def _extract_basic_info(self, packet: Dict) -> Dict:
        """从数据包中提取基本信息"""
        time_epoch = self._get_value(packet, ['_source', 'layers', 'frame', 'frame.time_epoch'])
        
        # 提取源地址
        src_addr = self._select_first(
            self._get_value(packet, ['_source', 'layers', 'ip', 'ip.src']),
            self._get_value(packet, ['_source', 'layers', 'eth', 'eth.src']),
            self._get_value(packet, ['_source', 'layers', 'zbee_nwk', 'zbee_nwk.src']),
            self._get_value(packet, ['_source', 'layers', 'wpan', 'wpan.src16'])
        )
        
        # 提取目标地址
        dst_addr = self._select_first(
            self._get_value(packet, ['_source', 'layers', 'ip', 'ip.dst']),
            self._get_value(packet, ['_source', 'layers', 'eth', 'eth.dst']),
            self._get_value(packet, ['_source', 'layers', 'zbee_nwk', 'zbee_nwk.dst']),
            self._get_value(packet, ['_source', 'layers', 'wpan', 'wpan.dst16'])
        )
        
        # 提取源端口
        src_port = self._select_first(
            self._get_value(packet, ['_source', 'layers', 'tcp', 'tcp.srcport']),
            self._get_value(packet, ['_source', 'layers', 'udp', 'udp.srcport'])
        )
        
        # 提取目标端口
        dst_port = self._select_first(
            self._get_value(packet, ['_source', 'layers', 'tcp', 'tcp.dstport']),
            self._get_value(packet, ['_source', 'layers', 'udp', 'udp.dstport'])
        )
        
        # 提取流ID
        stream_id = self._select_first(
            self._get_value(packet, ['_source', 'layers', 'tcp', 'tcp.stream']),
            self._get_value(packet, ['_source', 'layers', 'udp', 'udp.stream'])
        )
        
        return {
            'time': time_epoch,
            'src': (src_addr, src_port),
            'dst': (dst_addr, dst_port),
            'stream': stream_id
        }
    
    def _extract_split_position(self, start_bias: int, data_len: int, raw_layer: Dict) -> Dict:
        """提取字段分割位置信息，保留最细粒度的字段划分"""
        def traverse_field_info(json_node: Union[Dict, List], field_info: Dict, prefix: str = ''):
            """递归遍历字段信息 (修正以处理列表)"""
            if isinstance(json_node, dict):
                for field, field_data in json_node.items():
                    # 兼容 "Queries" 下的动态key
                    current_prefix = f"{prefix}.{field}" if prefix and not field.endswith(":") else field
                    traverse_field_info(field_data, field_info, prefix=current_prefix)
            elif isinstance(json_node, list):
                # 这是一个叶子节点
                field_info[prefix] = json_node

        # 收集所有字段信息
        field_list = {}
        traverse_field_info(raw_layer, field_list)

        # 提取所有字段的位置信息，包括层级关系
        all_positions = []
        max_bits = data_len * 8

        for field_name, field_info in field_list.items():
            # 跳过空字段或格式不正确的字段
            if len(field_info) < 3 or not isinstance(field_info[1], int) or not isinstance(field_info[2], int):
                continue
            
            # 跳过长度为0的字段
            if field_info[2] == 0:
                continue

            # 过滤不在当前协议数据范围内的字段
            if field_info[1] < start_bias:
                continue
            
            # 计算字节级别的相对位置
            byte_start = field_info[1] - start_bias
            byte_end = byte_start + field_info[2]
            
            # 确保不超出数据范围
            if byte_start >= data_len:
                continue
            
            # 如果结束位置超出范围，则截断
            if byte_end > data_len:
                byte_end = data_len

            # 解析位掩码
            bit_start, bit_end = self._parse_bitmask(field_info)

            if bit_start >= 0 and bit_end >= 0:
                # 位字段
                start_pos = byte_start * 8 + bit_start
                end_pos = byte_start * 8 + bit_end
                
                # 确保位掩码不超出字段范围
                if end_pos > byte_end * 8:
                    end_pos = byte_end * 8
            else:
                # 字节字段
                start_pos = byte_start * 8
                end_pos = byte_end * 8
            
            # 最终边界检查
            if start_pos >= max_bits or end_pos > max_bits or start_pos >= end_pos:
                continue
            
            # 存储字段信息，包括名称和层级深度（通过点的数量判断）
            depth = field_name.count('.')
            all_positions.append({
                'start': start_pos,
                'end': end_pos,
                'name': field_name,
                'depth': depth
            })
        
        # 按开始位置和结束位置排序
        all_positions.sort(key=lambda p: (p['start'], p['end']))
        
        # 找出最细粒度的字段划分
        # 将数据范围划分为不重叠的区间
        intervals = []
        
        # 如果没有有效字段，返回空结果
        if not all_positions:
            result = {
                'b_pos': [],
                'e_pos': [],
                'detailed_intervals': []
            }
            validation_result = self._validate_split_positions(result, data_len)
            result['validation'] = validation_result
            return result
        
        current_positions = sorted([(p['start'], 'start', i) for i, p in enumerate(all_positions)] + 
                                [(p['end'], 'end', i) for i, p in enumerate(all_positions)])
        
        # 使用扫描线算法找出所有不重叠的区间
        open_intervals = set()
        last_pos = None
        
        for pos, event_type, idx in current_positions:
            # 确保位置在有效范围内
            if pos > max_bits:
                continue
                
            # 如果位置变化且有开放的区间，则添加一个区间
            if last_pos is not None and pos > last_pos and open_intervals:
                # 在开放区间中找出深度最大的（最细粒度的）
                deepest_idx = max(open_intervals, key=lambda i: all_positions[i]['depth'])
                intervals.append((last_pos, pos, deepest_idx))
            
            # 更新开放区间集合
            if event_type == 'start':
                open_intervals.add(idx)
            else:
                if idx in open_intervals:
                    open_intervals.remove(idx)
            
            last_pos = pos
        
        # 从区间生成不重叠的 b_pos 和 e_pos
        b_pos = []
        e_pos = []
        
        for start, end, _ in intervals:
            if start not in b_pos:
                b_pos.append(start)
            if end not in e_pos:
                e_pos.append(end)
        
        # 确保 b_pos 和 e_pos 是有序的
        b_pos.sort()
        e_pos.sort()
        
        result = {
            'b_pos': b_pos,
            'e_pos': e_pos,
            'detailed_intervals': [(start, end, all_positions[idx]['name']) for start, end, idx in intervals]
        }

        # 添加位置验证
        validation_result = self._validate_split_positions(result, data_len)
        result['validation'] = validation_result

        return result
    
    def load_protocol_data(self, protocol: str) -> tuple:
        """加载指定协议的 JSON 和原始数据"""
        # 对于mbtcp协议，实际加载modbus的数据文件
        actual_protocol = 'modbus' if protocol.lower() == 'mbtcp' else protocol
        
        json_file = self.json_folder / f"{actual_protocol}.json"
        raw_file = self.jsonraw_folder / f"{actual_protocol}_raw.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        if not raw_file.exists():
            raise FileNotFoundError(f"Raw JSON file not found: {raw_file}")
        
        print(f"Loading {protocol} data from {actual_protocol} files...")
        
        try:
            with json_file.open('r', encoding='utf-8', errors='ignore') as f:
                json_data = json.load(f)
        except json.JSONDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            with json_file.open('r', encoding='latin-1') as f:
                json_data = json.load(f)
        
        try:
            with raw_file.open('r', encoding='utf-8', errors='ignore') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError:
            # Fallback to latin-1 if UTF-8 fails
            with raw_file.open('r', encoding='latin-1') as f:
                raw_data = json.load(f)
        
        print(f"Loaded {len(json_data)} packets for {protocol}")
        return json_data, raw_data

    
    def extract_protocol_splits(self, protocol: str) -> Generator[Dict, None, None]:
        """提取指定协议的数据分割信息"""
        if self.include_protocols and protocol.lower() not in self.include_protocols:
            print(f"Skipping protocol {protocol} (not in include list)")
            return
        
        try:
            json_data, raw_data = self.load_protocol_data(protocol)
        except FileNotFoundError as e:
            print(f"Error loading {protocol}: {e}")
            return
        
        # 对MODBUS协议使用特殊的优先级处理
        if protocol.lower() == 'modbus':
            yield from self._extract_modbus_with_priority(json_data, raw_data, protocol)
            return
        
        # 对MBTCP协议的特殊处理 - 从modbus数据中提取mbtcp层信息
        if protocol.lower() == 'mbtcp':
            yield from self._extract_mbtcp_from_modbus(json_data, raw_data)
            return
        
        # 其他协议使用统一的处理逻辑
        for pkt_id, (pkt, pkt_raw) in enumerate(zip(json_data, raw_data)):
            info = self._extract_basic_info(pkt)
            info['pkt_id'] = pkt_id
            info['protocol'] = protocol
            
            layers = pkt_raw.get('_source', {}).get('layers', {})
            
            if protocol in layers and f"{protocol}_raw" in layers:
                layer_raw_data = layers[f"{protocol}_raw"]
                
                if isinstance(layer_raw_data, list) and len(layer_raw_data) >= 3:
                    data_len = layer_raw_data[2]
                    
                    # 只提取 s7comm 协议中长度为20的数据包
                    if protocol.lower() == 's7comm' and data_len != 20:
                        continue

                    info['data'] = layer_raw_data[0]
                    info['data_len'] = data_len
                    
                    info['split_pos'] = self._extract_split_position(
                        layer_raw_data[1],
                        layer_raw_data[2],
                        layers[protocol]
                    )
                    
                    yield info

    def _extract_mbtcp_from_modbus(self, json_data, raw_data):
        """从modbus数据中提取mbtcp协议信息（包含完整的mbtcp+modbus数据）"""
        for pkt_id, (pkt, pkt_raw) in enumerate(zip(json_data, raw_data)):
            layers = pkt_raw.get('_source', {}).get('layers', {})
            
            # 检查是否同时包含mbtcp和modbus层
            if 'mbtcp' in layers and 'mbtcp_raw' in layers and 'modbus' in layers and 'modbus_raw' in layers:
                mbtcp_raw_data = layers['mbtcp_raw']
                
                if isinstance(mbtcp_raw_data, list) and len(mbtcp_raw_data) >= 3:
                    # 提取基本信息
                    info = self._extract_basic_info(pkt)
                    info['pkt_id'] = pkt_id
                    info['protocol'] = 'mbtcp'
                    
                    # 使用完整的mbtcp数据（包含内嵌的modbus）
                    info['data'] = mbtcp_raw_data[0]
                    info['data_len'] = mbtcp_raw_data[2]
                    
                    # 为mbtcp协议提取详细的字段分割（包括mbtcp头部字段和内嵌的modbus字段）
                    info['split_pos'] = self._extract_mbtcp_split_position(
                        mbtcp_raw_data[1],
                        mbtcp_raw_data[2],
                        layers
                    )
                    
                    yield info

    def _extract_mbtcp_split_position(self, start_bias: int, data_len: int, layers: Dict) -> Dict:
        """为MBTCP协议提取详细的字段分割位置（包括mbtcp头和modbus数据）"""
        def traverse_field_info(json_node: Union[Dict, List], field_info: Dict, prefix: str = ''):
            """递归遍历字段信息"""
            if isinstance(json_node, dict):
                for field, field_data in json_node.items():
                    current_prefix = f"{prefix}.{field}" if prefix and not field.endswith(":") else field
                    traverse_field_info(field_data, field_info, prefix=current_prefix)
            elif isinstance(json_node, list):
                field_info[prefix] = json_node

        # 收集所有字段信息（包括mbtcp和modbus层）
        field_list = {}
        
        # 添加mbtcp层字段
        if 'mbtcp' in layers:
            traverse_field_info(layers['mbtcp'], field_list, prefix='mbtcp')
        
        # 添加modbus层字段
        if 'modbus' in layers:
            traverse_field_info(layers['modbus'], field_list, prefix='modbus')

        # 提取所有字段的位置信息
        all_positions = []
        max_bits = data_len * 8

        for field_name, field_info in field_list.items():
            # 跳过空字段或格式不正确的字段
            if len(field_info) < 3 or not isinstance(field_info[1], int) or not isinstance(field_info[2], int):
                continue
            
            # 跳过长度为0的字段
            if field_info[2] == 0:
                continue

            # 过滤不在当前协议数据范围内的字段
            if field_info[1] < start_bias:
                continue
            
            # 计算字节级别的相对位置
            byte_start = field_info[1] - start_bias
            byte_end = byte_start + field_info[2]
            
            # 确保不超出数据范围
            if byte_start >= data_len:
                continue
            
            # 如果结束位置超出范围，则截断
            if byte_end > data_len:
                byte_end = data_len

            # 解析位掩码
            bit_start, bit_end = self._parse_bitmask(field_info)

            if bit_start >= 0 and bit_end >= 0:
                # 位字段
                start_pos = byte_start * 8 + bit_start
                end_pos = byte_start * 8 + bit_end
                
                # 确保位掩码不超出字段范围
                if end_pos > byte_end * 8:
                    end_pos = byte_end * 8
            else:
                # 字节字段
                start_pos = byte_start * 8
                end_pos = byte_end * 8
            
            # 最终边界检查
            if start_pos >= max_bits or end_pos > max_bits or start_pos >= end_pos:
                continue
            
            # 存储字段信息，包括名称和层级深度
            depth = field_name.count('.')
            all_positions.append({
                'start': start_pos,
                'end': end_pos,
                'name': field_name,
                'depth': depth
            })
        
        # 按开始位置和结束位置排序
        all_positions.sort(key=lambda p: (p['start'], p['end']))
        
        # 找出最细粒度的字段划分
        intervals = []
        
        # 如果没有有效字段，返回空结果
        if not all_positions:
            result = {
                'b_pos': [],
                'e_pos': [],
                'detailed_intervals': []
            }
            validation_result = self._validate_split_positions(result, data_len)
            result['validation'] = validation_result
            return result
        
        current_positions = sorted([(p['start'], 'start', i) for i, p in enumerate(all_positions)] + 
                                [(p['end'], 'end', i) for i, p in enumerate(all_positions)])
        
        # 使用扫描线算法找出所有不重叠的区间
        open_intervals = set()
        last_pos = None
        
        for pos, event_type, idx in current_positions:
            # 确保位置在有效范围内
            if pos > max_bits:
                continue
                
            # 如果位置变化且有开放的区间，则添加一个区间
            if last_pos is not None and pos > last_pos and open_intervals:
                # 在开放区间中找出深度最大的（最细粒度的）
                deepest_idx = max(open_intervals, key=lambda i: all_positions[i]['depth'])
                intervals.append((last_pos, pos, deepest_idx))
            
            # 更新开放区间集合
            if event_type == 'start':
                open_intervals.add(idx)
            else:
                if idx in open_intervals:
                    open_intervals.remove(idx)
            
            last_pos = pos
        
        # 从区间生成不重叠的 b_pos 和 e_pos
        b_pos = []
        e_pos = []
        
        for start, end, _ in intervals:
            if start not in b_pos:
                b_pos.append(start)
            if end not in e_pos:
                e_pos.append(end)
        
        # 确保 b_pos 和 e_pos 是有序的
        b_pos.sort()
        e_pos.sort()
        
        result = {
            'b_pos': b_pos,
            'e_pos': e_pos,
            'detailed_intervals': [(start, end, all_positions[idx]['name']) for start, end, idx in intervals]
        }

        # 添加位置验证
        validation_result = self._validate_split_positions(result, data_len)
        result['validation'] = validation_result

        return result
       
    def _extract_modbus_with_priority(self, json_data, raw_data, protocol):
        """为MODBUS协议实现优先级筛选：优先要5字节长度04开头的包，不足时补充02开头的5字节包"""
        # 收集所有5字节长度的MODBUS包，按功能码分类
        packets_04_5bytes = []
        packets_02_5bytes = []
        
        for pkt_id, (pkt, pkt_raw) in enumerate(zip(json_data, raw_data)):
            layers = pkt_raw.get('_source', {}).get('layers', {})
            
            if protocol in layers and f"{protocol}_raw" in layers:
                layer_raw_data = layers[f"{protocol}_raw"]
                
                if isinstance(layer_raw_data, list) and len(layer_raw_data) >= 3:
                    data_len = layer_raw_data[2]
                    data = layer_raw_data[0]
                    
                    # 只处理5字节长度的包
                    if data_len != 5:
                        continue
                    
                    # 提取基本信息
                    info = self._extract_basic_info(pkt)
                    info['pkt_id'] = pkt_id
                    info['protocol'] = protocol
                    info['data'] = data
                    info['data_len'] = data_len
                    info['split_pos'] = self._extract_split_position(
                        layer_raw_data[1],
                        layer_raw_data[2],
                        layers[protocol]
                    )
                    
                    # 根据功能码分类（数据的第一个字节）
                    if len(data) >= 2:
                        function_code = data[:2].lower()
                        
                        if function_code == '04':
                            packets_04_5bytes.append(info)
                        elif function_code == '02':
                            packets_02_5bytes.append(info)
        
        print(f"MODBUS包分布 (5字节): 04功能码: {len(packets_04_5bytes)} 个, 02功能码: {len(packets_02_5bytes)} 个")
        
        # 优先输出04功能码的包
        for packet_info in packets_04_5bytes:
            yield packet_info
        
        # 如果04功能码的包不足，补充02功能码的包
        for packet_info in packets_02_5bytes:
            yield packet_info
    
    def extract_all_splits(self) -> Generator[Dict, None, None]:
        """提取所有协议的数据分割信息"""
        protocols_to_process = self.available_protocols
        
        if self.include_protocols:
            protocols_to_process = [p for p in protocols_to_process 
                                  if p.lower() in self.include_protocols]
        
        print(f"Processing protocols: {protocols_to_process}")
        
        for protocol in protocols_to_process:
            if protocol == 'merged':  # 跳过合并文件
                continue
            
            print(f"\nProcessing protocol: {protocol}")
            yield from self.extract_protocol_splits(protocol)
    
    def get_protocol_stats(self) -> Dict[str, int]:
        """获取各协议的数据包统计"""
        stats = {}
        for protocol in self.available_protocols:
            if protocol == 'merged':
                continue
            try:
                json_data, _ = self.load_protocol_data(protocol)
                stats[protocol] = len(json_data)
            except FileNotFoundError:
                stats[protocol] = 0
        return stats

    def save_protocol_data(self, output_dir: Union[str, Path], max_packets_per_protocol: Optional[int] = None, 
                          save_used_pcaps: bool = True, pcap_source_dir: Union[str, Path] = None) -> Dict[str, int]:
        """
        保存所有协议的数据包信息到JSON文件，并可选择保存实际使用的pcap包
        
        Args:
            output_dir: 输出目录路径
            max_packets_per_protocol: 每个协议最多保存的数据包数量，None表示保存所有
            save_used_pcaps: 是否保存实际使用的pcap包
            pcap_source_dir: 原始pcap文件所在目录，如果save_used_pcaps为True则必须提供
        
        Returns:
            保存的数据包统计信息
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 创建用于保存实际使用pcap的目录
        if save_used_pcaps:
            if pcap_source_dir is None:
                raise ValueError("如果要保存实际使用的pcap包，必须提供pcap_source_dir参数")
            
            pcap_source_path = Path(pcap_source_dir)
            if not pcap_source_path.exists():
                raise FileNotFoundError(f"PCAP源目录不存在: {pcap_source_path}")
            
            real_pcap_dir = output_path / "real_used_pcaps"
            real_pcap_dir.mkdir(parents=True, exist_ok=True)
        
        # 用于统计保存的数据包数量
        saved_stats = {}
        
        # 处理每个协议
        for protocol in self.available_protocols:
            if protocol == 'merged':  # 跳过合并文件
                continue
            
            if self.include_protocols and protocol.lower() not in self.include_protocols:
                continue
            
            print(f"\n处理协议: {protocol}")
            
            # 收集该协议的所有数据包
            protocol_packets = []
            packet_count = 0
            used_packet_ids = []  # 记录实际使用的包ID
            
            try:
                # 提取协议数据
                for packet_info in self.extract_protocol_splits(protocol):
                    # 如果达到最大数量限制，则停止
                    if max_packets_per_protocol and packet_count >= max_packets_per_protocol:
                        break
                    
                    # 记录使用的包ID
                    used_packet_ids.append(packet_info['pkt_id'])
                    
                    # 提取分割位置
                    b_pos = packet_info.get('split_pos', {}).get('b_pos', [])
                    e_pos = packet_info.get('split_pos', {}).get('e_pos', [])
                    data = packet_info.get('data', '')
                    
                    # 生成分割后的数据
                    split_data = ""
                    if data and b_pos and e_pos and len(b_pos) == len(e_pos):
                        # 确保位置是按顺序排列的
                        positions = sorted(zip(b_pos, e_pos))
                        
                        # 将数据按位置分割
                        for i, (start_bit, end_bit) in enumerate(positions):
                            # 转换为字节位置
                            start_byte = start_bit // 8
                            end_byte = (end_bit + 7) // 8  # 向上取整到字节边界
                            
                            # 确保不超出数据范围
                            if start_byte >= len(data) // 2:
                                continue
                            if end_byte > len(data) // 2:
                                end_byte = len(data) // 2
                            
                            # 提取该字段的数据
                            field_data = data[start_byte*2:end_byte*2]
                            
                            # 添加到分割后的数据中，用空格分隔
                            if i > 0:
                                split_data += " "
                            split_data += field_data
                    
                    # 提取关键信息
                    formatted_packet = {
                        'protocol': protocol,
                        'packet_id': packet_info['pkt_id'],
                        'time': packet_info.get('time'),
                        'src_addr': packet_info.get('src', [None, None])[0],
                        'src_port': packet_info.get('src', [None, None])[1],
                        'dst_addr': packet_info.get('dst', [None, None])[0],
                        'dst_port': packet_info.get('dst', [None, None])[1],
                        'stream_id': packet_info.get('stream'),
                        'data': packet_info.get('data'),
                        'data_len': packet_info.get('data_len'),
                        'split_positions': {
                            'b_pos': packet_info.get('split_pos', {}).get('b_pos', []),
                            'e_pos': packet_info.get('split_pos', {}).get('e_pos', []),
                        }
                    }
                    
                    protocol_packets.append(formatted_packet)
                    packet_count += 1
                
                # 保存该协议实际使用的包ID记录
                self.used_packet_ids[protocol] = used_packet_ids
                
                # 将该协议的数据保存为单独的JSON文件
                if protocol_packets:
                    protocol_output = output_path / f"{protocol}_data.json"
                    with open(protocol_output, 'w', encoding='utf-8') as f:
                        # 先转换为字符串，使用标准缩进
                        json_str = json.dumps(protocol_packets, ensure_ascii=False, indent=2)
                        
                        # 使用正则表达式将b_pos和e_pos数组从多行变为单行，并移除多余空格
                        json_str = re.sub(
                            r'("(?:b_pos|e_pos)":\s*\[)([\d\s,]+)(\])', 
                            lambda m: m.group(1) + ' ' + re.sub(r'\s+', ' ', m.group(2).strip()) + ' ' + m.group(3),
                            json_str
                        )
                        
                        # 写入文件
                        f.write(json_str)
                    
                    saved_stats[protocol] = len(protocol_packets)
                    print(f"已保存 {len(protocol_packets)} 个 {protocol} 数据包到 {protocol_output}")
                    
                    # 保存实际使用的pcap包
                    if save_used_pcaps and used_packet_ids:
                        # 对于mbtcp协议，也使用modbus的pcap文件，但不单独生成mbtcp.pcap
                        source_protocol = 'modbus' if protocol.lower() == 'mbtcp' else protocol
                        self._save_used_pcap_packets(
                            source_protocol, 
                            used_packet_ids, 
                            pcap_source_path, 
                            real_pcap_dir
                        )
                        
                else:
                    print(f"未找到任何 {protocol} 数据包")
                    saved_stats[protocol] = 0
                
            except Exception as e:
                print(f"处理协议 {protocol} 时出错: {e}")
                saved_stats[protocol] = 0
        
        # 保存汇总统计信息
        stats_output = output_path / "protocol_stats.json"
        with open(stats_output, 'w', encoding='utf-8') as f:
            stats_data = {
                'total_packets': sum(saved_stats.values()),
                'protocols': saved_stats
            }
            
            # 如果保存了实际使用的pcap包，添加相关信息
            if save_used_pcaps:
                stats_data['used_packet_ids'] = self.used_packet_ids
                stats_data['real_used_pcaps_dir'] = str(real_pcap_dir)
            
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n已保存所有协议统计信息到 {stats_output}")
        print(f"总共保存了 {sum(saved_stats.values())} 个数据包")
        
        if save_used_pcaps:
            print(f"实际使用的pcap包已保存到: {real_pcap_dir}")
        
        return saved_stats
    
    def _save_used_pcap_packets(self, protocol: str, used_packet_ids: List[int], 
                               pcap_source_dir: Path, output_dir: Path):
        """
        保存指定协议实际使用的pcap包
        
        Args:
            protocol: 协议名称
            used_packet_ids: 实际使用的包ID列表（从0开始）
            pcap_source_dir: 原始pcap文件目录
            output_dir: 输出目录
        """
        try:
            source_pcap = pcap_source_dir / f"{protocol}.pcap"
            if not source_pcap.exists():
                print(f"⚠️  警告: 源pcap文件不存在: {source_pcap}")
                return
            
            output_pcap = output_dir / f"{protocol}.pcap"
            
            # 创建包过滤表达式，tshark使用1开始的包编号
            # 将0开始的包ID转换为1开始的包编号
            packet_numbers = [str(pkt_id + 1) for pkt_id in used_packet_ids]
            
            # 构建tshark命令来提取指定的包
            # 使用frame.number来过滤特定的包
            filter_expr = "frame.number==" + " or frame.number==".join(packet_numbers)
            
            command = [
                "tshark", 
                "-r", str(source_pcap),
                "-Y", filter_expr,
                "-w", str(output_pcap)
            ]
            
            print(f"正在提取 {protocol} 协议的 {len(used_packet_ids)} 个实际使用的包...")
            
            # 执行命令
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # 验证输出文件
            if output_pcap.exists() and output_pcap.stat().st_size > 0:
                print(f"✅ 成功保存 {protocol} 实际使用的pcap包到: {output_pcap}")
                
                # 验证包数量
                verify_command = ["tshark", "-r", str(output_pcap), "-c", "999999"]
                verify_result = subprocess.run(
                    verify_command, 
                    capture_output=True, 
                    text=True
                )
                actual_count = len(verify_result.stdout.strip().split('\n')) if verify_result.stdout.strip() else 0
                print(f"   验证: 预期 {len(used_packet_ids)} 个包，实际保存 {actual_count} 个包")
                
                if actual_count != len(used_packet_ids):
                    print(f"⚠️  警告: 包数量不匹配！")
            else:
                print(f"❌ 保存失败: {output_pcap}")
                
        except subprocess.CalledProcessError as e:
            print(f"❌ 提取 {protocol} pcap包时出错: {e}")
            print(f"   命令输出: {e.stdout}")
            print(f"   错误输出: {e.stderr}")
        except Exception as e:
            print(f"❌ 处理 {protocol} pcap包时出错: {e}")


# 使用示例
if __name__ == "__main__":
    # 初始化处理器
    handler = ProtocolJsonHandler(
        json_folder="./data/processed/json",
        jsonraw_folder="./data/processed/jsonraw",
        include_protocols=None  # 处理所有协议
    )
    
    # print("Available protocols:", handler.available_protocols)
    
    # # 获取统计信息
    # stats = handler.get_protocol_stats()
    # print("Protocol stats:", stats)
    
    # 保存所有协议数据到指定目录，并保存实际使用的pcap包
    output_dir = "./data/formatted"
    saved_stats = handler.save_protocol_data(
        output_dir=output_dir,
        max_packets_per_protocol=2000,  # 每个协议最多保存2000个数据包，设为None则保存所有
        save_used_pcaps=True,  # 启用保存实际使用的pcap包
        pcap_source_dir="./data/protocol"  # 原始pcap文件目录
    )
    
    # 用于跟踪每个协议是否已打印第一个包
    printed_first = {}
    # 收集有问题的数据包
    problem_packets = []
    
    # 处理所有协议的所有数据包
    total_processed = 0
    total_valid = 0
    total_invalid = 0
    
    print("\n开始处理所有协议...")
    
    for split_info in handler.extract_all_splits():
        protocol = split_info['protocol']
        pkt_id = split_info['pkt_id']
        total_processed += 1
        
        # 检查验证结果
        validation = split_info.get('split_pos', {}).get('validation', {})
        is_valid = validation.get('is_valid', True)
        has_warnings = bool(validation.get('warnings', []))
        
        if is_valid:
            total_valid += 1
        else:
            total_invalid += 1
            
        # 如果是该协议的第一个包，或者包含错误/警告，则收集信息
        if protocol not in printed_first:
            printed_first[protocol] = True
            
            print(f"\n{'='*60}")
            print(f"Protocol: {protocol} - Packet #{pkt_id+1}")
            print(f"{'='*60}")
            
            if validation:
                print(f"🔍 VALIDATION RESULTS:")
                print(f"   Valid: {is_valid}")
                if validation.get('errors'):
                    print(f"   ❌ Errors: {validation['errors']}")
                if validation.get('warnings'):
                    print(f"   ⚠️  Warnings: {validation['warnings']}")
                if validation.get('stats'):
                    stats = validation['stats']
                    print(f"   📊 Stats: {stats}")
                print()
            
            # 使用 json.dumps 格式化输出完整信息
            print(json.dumps(split_info, indent=2, ensure_ascii=False))
        
        # 如果数据包有问题，收集它
        if not is_valid or has_warnings:
            problem_info = {
                'protocol': protocol,
                'pkt_id': pkt_id,
                'is_valid': is_valid,
                'errors': validation.get('errors', []),
                'warnings': validation.get('warnings', []),
                'stats': validation.get('stats', {})
            }
            problem_packets.append(problem_info)
    
    # 打印汇总信息
    print(f"\n{'='*60}")
    print(f"处理完成! 总结:")
    print(f"{'='*60}")
    print(f"总处理数据包: {total_processed}")
    print(f"有效数据包: {total_valid}")
    print(f"无效数据包: {total_invalid}")
    print(f"有问题的数据包: {len(problem_packets)}")
    
    # 打印所有有问题的数据包
    if problem_packets:
        print(f"\n{'='*60}")
        print(f"有问题的数据包详情:")
        print(f"{'='*60}")
        
        for i, problem in enumerate(problem_packets):
            print(f"\n问题 #{i+1}:")
            print(f"协议: {problem['protocol']}, 包ID: {problem['pkt_id']}")
            print(f"有效性: {'✅ 有效但有警告' if problem['is_valid'] else '❌ 无效'}")
            
            if problem['errors']:
                print(f"错误: {problem['errors']}")
            if problem['warnings']:
                print(f"警告: {problem['warnings']}")
            
            # 打印关键统计信息
            if 'coverage_percentage' in problem['stats']:
                print(f"覆盖率: {problem['stats'].get('coverage_percentage')}%")
            if 'b_pos_count' in problem['stats'] and 'e_pos_count' in problem['stats']:
                print(f"位置数量: 开始={problem['stats'].get('b_pos_count')}, 结束={problem['stats'].get('e_pos_count')}")
