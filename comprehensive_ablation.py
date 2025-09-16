#!/usr/bin/env python3
"""
综合消融实验脚本
测试结构特征和模型组件的影响
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import re
from config import TRAINING_CONFIG

# 定义综合消融实验配置
COMPREHENSIVE_ABLATION_CONFIGS = [
    {
        'name': 'baseline',
        'description': 'baseline',
        'use_structure_features': False,
        'enable_structure_guidance': False,
        'structure_features': [],
        'enable_ipc': False
    },
    {
        'name': 'without_guidance',
        'description': 'without_guidance',
        'use_structure_features': True,
        'enable_structure_guidance': False,
        'structure_features': ['abc', 'entropy'],
        'enable_ipc': True
    },
    {
        'name': 'full_model',
        'description': 'full_model',
        'use_structure_features': True,
        'enable_structure_guidance': True,
        'structure_features': ['abc', 'entropy'],
        'enable_ipc': True
    },
    {
        'name': 'withoutipc',
        'description': 'withoutipc',
        'use_structure_features': True,
        'enable_structure_guidance': True,
        'structure_features': ['abc', 'entropy'],
        'enable_ipc': False
    },
    {
        'name': 'without_stru',
        'description': 'without_stru',
        'use_structure_features': False,
        'enable_structure_guidance': False,
        'structure_features': [],
        'enable_ipc': True
    }
]

def backup_config():
    """备份原始配置文件"""
    shutil.copy('config.py', 'config.py.backup')
    print("✓ 已备份原始配置文件")

def restore_config():
    """恢复原始配置文件"""
    if os.path.exists('config.py.backup'):
        shutil.copy('config.py.backup', 'config.py')
        print("✓ 已恢复原始配置文件")

def modify_config(config):
    """修改配置文件中的模型配置"""
    with open('config.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换 use_structure_features
    pattern_use = r"('use_structure_features':\s*)(True|False)"
    replacement_use = f"\\g<1>{config['use_structure_features']}"
    content = re.sub(pattern_use, replacement_use, content)
    
    # 替换 enable_structure_guidance
    pattern_guidance = r"('enable_structure_guidance':\s*)(True|False)"
    replacement_guidance = f"\\g<1>{config['enable_structure_guidance']}"
    content = re.sub(pattern_guidance, replacement_guidance, content)
    
    # 替换 structure_features
    pattern_features = r"('structure_features':\s*)\[[^\]]*\]"
    replacement_features = f"\\g<1>{config['structure_features']}"
    content = re.sub(pattern_features, replacement_features, content)
    
    # 替换 enable_ipc
    pattern_cross = r"('enable_ipc':\s*)(True|False)"
    replacement_cross = f"\\g<1>{config['enable_ipc']}"
    content = re.sub(pattern_cross, replacement_cross, content)
    
    with open('config.py', 'w', encoding='utf-8') as f:
        f.write(content)

def run_experiment(config):
    """运行单个消融实验"""
    print(f"\n{'='*80}")
    print(f"开始实验: {config['description']}")
    print(f"配置详情:")
    print(f"  - 使用结构特征: {config['use_structure_features']}")
    print(f"  - 结构引导机制: {config['enable_structure_guidance']}")
    print(f"  - 结构特征种类: {config['structure_features']}")
    print(f"  - 包间协作: {config['enable_ipc']}")
    print(f"{'='*80}")
    
    # 修改配置
    modify_config(config)
    
    # 创建实验结果目录
    result_dir = Path(f"ablation_results/comprehensive/{config['name']}")
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # 运行训练脚本
    try:
        # 重定向输出到文件
        log_file = result_dir / "training.log"
        
        print(f"开始训练... 日志将保存到: {log_file}")
        
        with open(log_file, 'w', encoding='utf-8') as f:
            # 运行训练脚本
            process = subprocess.Popen(
                [sys.executable, 'train.py'],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出并保存到文件
            for line in iter(process.stdout.readline, ''):
                print(line.rstrip())  # 显示到控制台
                f.write(line)         # 保存到文件
                f.flush()
            
            process.stdout.close()
            return_code = process.wait()
            
        if return_code == 0:
            print(f"✓ 实验 {config['name']} 完成")
            
            # 保存实验配置
            config_file = result_dir / "experiment_config.json"
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
                
        else:
            print(f"✗ 实验 {config['name']} 失败，返回码: {return_code}")
            
    except Exception as e:
        print(f"✗ 实验 {config['name']} 出错: {e}")

def extract_results():
    """提取所有实验结果"""
    print(f"\n{'='*80}")
    print("提取实验结果...")
    print(f"{'='*80}")
    
    results = []
    
    for config in COMPREHENSIVE_ABLATION_CONFIGS:
        result_dir = Path(f"ablation_results/comprehensive/{config['name']}")
        log_file = result_dir / "training.log"
        
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                
                # 提取关键指标
                import re
                
                # 匹配模式：留一法交叉验证最终结果汇总之后的平均结果行
                pattern = r"留一法交叉验证最终结果汇总.*?平均\s*-\s*F1:\s*([\d.]+)±[\d.]+,\s*Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+)"
                match = re.search(pattern, log_content, re.DOTALL)
                
                if match:
                    f1_score = float(match.group(1))
                    precision = float(match.group(2))
                    recall = float(match.group(3))
                    
                    # 同时提取各协议的详细结果
                    protocol_results = {}
                    
                    # 查找各协议结果行
                    protocol_pattern = r"(\w+)\s*-\s*F1:\s*([\d.]+),\s*Precision:\s*([\d.]+),\s*Recall:\s*([\d.]+)"
                    protocol_matches = re.findall(protocol_pattern, log_content)
                    
                    for protocol_match in protocol_matches:
                        protocol_name = protocol_match[0].strip()
                        if protocol_name != '平均':  # 排除平均值行
                            protocol_results[protocol_name] = {
                                'f1': float(protocol_match[1]),
                                'precision': float(protocol_match[2]),
                                'recall': float(protocol_match[3])
                            }
                    
                    results.append({
                        'name': config['name'],
                        'description': config['description'],
                        'use_structure_features': config['use_structure_features'],
                        'enable_structure_guidance': config['enable_structure_guidance'],
                        'structure_features': config['structure_features'],
                        'enable_ipc': config['enable_ipc'],
                        'f1_score': f1_score,
                        'precision': precision,
                        'recall': recall,
                        'protocol_results': protocol_results
                    })
                    
                    print(f"{config['description']}: F1={f1_score:.4f}, P={precision:.4f}, R={recall:.4f}")
                    print(f"  协议数量: {len(protocol_results)}")
                else:
                    print(f"{config['description']}: 无法提取结果")
                    # 尝试备用模式
                    backup_pattern = r"测试结果.*?F1:\s*([\d.]+).*?精确率:\s*([\d.]+).*?召回率:\s*([\d.]+)"
                    backup_match = re.search(backup_pattern, log_content, re.DOTALL)
                    if backup_match:
                        print(f"  发现备用格式结果")
                        f1_score = float(backup_match.group(1))
                        precision = float(backup_match.group(2))
                        recall = float(backup_match.group(3))
                        
                        results.append({
                            'name': config['name'],
                            'description': config['description'],
                            'use_structure_features': config['use_structure_features'],
                            'enable_structure_guidance': config['enable_structure_guidance'],
                            'structure_features': config['structure_features'],
                            'enable_ipc': config['enable_ipc'],
                            'f1_score': f1_score,
                            'precision': precision,
                            'recall': recall,
                            'protocol_results': {}
                        })
                        print(f"{config['description']}: F1={f1_score:.4f}, P={precision:.4f}, R={recall:.4f} (备用格式)")
                    
            except Exception as e:
                print(f"{config['description']}: 提取结果出错 - {e}")
        else:
            print(f"{config['description']}: 日志文件不存在")
    
    # 保存汇总结果
    epochs = TRAINING_CONFIG['num_epochs']
    summary_file = Path(f"ablation_results/comprehensive/summary_epoch{epochs}.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'experiment_type': 'comprehensive_ablation',
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 结果汇总已保存到: {summary_file}")
    
    # 分析和展示结果
    if results:
        analyze_results(results)

def analyze_results(results):
    """分析实验结果"""
    print(f"\n{'='*80}")
    print("综合消融实验结果分析")
    print(f"{'='*80}")
    
    # 按F1分数排序
    sorted_results = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    
    print(f"{'排名':<4} {'实验名称':<35} {'F1分数':<8} {'精确率':<8} {'召回率':<8}")
    print("-" * 90)
    
    for i, result in enumerate(sorted_results, 1):
        print(f"{i:<4} {result['description']:<35} {result['f1_score']:<8.4f} {result['precision']:<8.4f} {result['recall']:<8.4f}")
    
    # 找到完整模型作为基准
    full_model_result = next((r for r in results if r['name'] == 'full_model'), None)
    if full_model_result:
        baseline_f1 = full_model_result['f1_score']
        
        print(f"\n{'='*80}")
        print("相对于完整模型的性能变化")
        print(f"{'='*80}")
        print(f"基准（完整模型）F1分数: {baseline_f1:.4f}")
        print()
        
        for result in sorted_results:
            if result['name'] != 'full_model':
                diff = result['f1_score'] - baseline_f1
                percentage = (diff / baseline_f1) * 100
                status = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
                print(f"{status} {result['description']:<35}: {diff:+.4f} ({percentage:+.2f}%)")
    
    # 分析各组件的贡献
    print(f"\n{'='*80}")
    print("组件贡献分析")
    print(f"{'='*80}")
    
    # 查找关键比较对
    baseline = next((r for r in results if r['name'] == 'baseline'), None)
    struct_no_guide = next((r for r in results if r['name'] == 'structure_no_guidance_with_cross'), None)
    full = next((r for r in results if r['name'] == 'full_model'), None)
    struct_guide_no_cross = next((r for r in results if r['name'] == 'structure_guidance_no_cross'), None)
    
    if all([baseline, struct_no_guide, full, struct_guide_no_cross]):
        print("� 三大创新模块的价值验证:")
        print()
        
        # 1. 结构信息的价值：比较基线模型 vs 有结构信息但无引导
        if baseline and struct_no_guide:
            diff = struct_no_guide['f1_score'] - baseline['f1_score']
            print(f"🔍 创新点1 - 结构信息的价值:")
            print(f"   基线模型 vs 结构信息(无引导): {diff:+.4f}")
        
        # 2. 结构引导的价值：比较有结构无引导 vs 完整模型
        if struct_no_guide and full:
            diff = full['f1_score'] - struct_no_guide['f1_score']
            print(f"\n🎯 创新点2 - 结构引导机制的价值:")
            print(f"   结构信息(无引导) vs 完整模型: {diff:+.4f}")
        
        # 3. 包间协作的价值：比较有结构引导无协作 vs 完整模型
        if struct_guide_no_cross and full:
            diff = full['f1_score'] - struct_guide_no_cross['f1_score']
            print(f"\n🤝 创新点3 - 包间协作的价值:")
            print(f"   结构引导(无协作) vs 完整模型: {diff:+.4f}")
        
        # 总体提升
        if baseline and full:
            total_improvement = full['f1_score'] - baseline['f1_score']
            print(f"\n🚀 总体创新价值:")
            print(f"   基线模型 → 完整模型: {total_improvement:+.4f}")
            
            # 各模块独立贡献计算
            if struct_no_guide and struct_guide_no_cross:
                struct_info_contrib = struct_no_guide['f1_score'] - baseline['f1_score']
                guidance_contrib = full['f1_score'] - struct_no_guide['f1_score']
                cross_contrib = full['f1_score'] - struct_guide_no_cross['f1_score']
                
                print(f"\n📈 各模块独立贡献:")
                print(f"   结构信息: {struct_info_contrib:.4f}")
                print(f"   结构引导: {guidance_contrib:.4f}")
                print(f"   包间协作: {cross_contrib:.4f}")
                
                # 计算贡献占比
                if total_improvement > 0:
                    print(f"\n📊 贡献占比:")
                    print(f"   结构信息: {(struct_info_contrib/total_improvement)*100:.1f}%")
                    print(f"   结构引导: {(guidance_contrib/total_improvement)*100:.1f}%")
                    print(f"   包间协作: {(cross_contrib/total_improvement)*100:.1f}%")

def main():
    """主函数"""
    print("综合消融实验开始")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"将测试 {len(COMPREHENSIVE_ABLATION_CONFIGS)} 种配置:")
    
    for i, config in enumerate(COMPREHENSIVE_ABLATION_CONFIGS, 1):
        print(f"  {i}. {config['description']}")
    
    # 确认是否继续
    response = input(f"\n是否继续? (y/N): ").strip().lower()
    if response != 'y':
        print("实验取消")
        return
    
    try:
        # 备份配置
        backup_config()
        
        # 运行所有实验
        for i, config in enumerate(COMPREHENSIVE_ABLATION_CONFIGS, 1):
            print(f"\n进度: {i}/{len(COMPREHENSIVE_ABLATION_CONFIGS)}")
            run_experiment(config)
        
        # 恢复配置
        restore_config()
        
        # 提取并汇总结果
        extract_results()
        
        print(f"\n{'='*80}")
        print("所有实验完成!")
        print(f"{'='*80}")
        
    except KeyboardInterrupt:
        print("\n实验被用户中断")
        restore_config()
    except Exception as e:
        print(f"\n实验出错: {e}")
        restore_config()

if __name__ == "__main__":
    main()
