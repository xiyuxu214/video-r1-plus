# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LaTeX Table Generator

Automatically generates LaTeX tables from experimental results:
1. Main results table (vs SOTA)
2. Ablation study table
3. Robustness analysis table
4. Computational efficiency comparison table
5. Selection decision analysis table

Outputs ready-to-paste LaTeX code.
"""

import json
import argparse
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np


class LaTeXTableGenerator:
    """
    LaTeX表格生成器
    
    根据实验结果自动生成各种类型的LaTeX表格。
    """
    
    def __init__(self, decimal_places: int = 2):
        self.decimal_places = decimal_places
    
    def format_number(self, value: float, unit: str = "", bold: bool = False) -> str:
        """
        格式化数字为LaTeX格式
        
        Args:
            value: 数值
            unit: 单位（如"%", "ms"等）
            bold: 是否加粗
        
        Returns:
            格式化后的字符串
        """
        if value is None or np.isnan(value):
            return "---"
        
        formatted = f"{value:.{self.decimal_places}f}"
        if bold:
            formatted = f"\\textbf{{{formatted}}}"
        if unit:
            formatted = f"{formatted}{unit}"
        
        return formatted
    
    def escape_latex(self, text: str) -> str:
        """转义LaTeX特殊字符"""
        special_chars = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '^': r'\^{}',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '\\': r'\textbackslash{}',
        }
        for char, replacement in special_chars.items():
            text = text.replace(char, replacement)
        return text
    
    def generate_main_results_table(
        self,
        results: Dict[str, Dict[str, float]],
        baselines: Optional[Dict[str, Dict[str, float]]] = None,
        benchmarks: Optional[List[str]] = None,
    ) -> str:
        """
        生成主结果表（与SOTA对比）
        
        Args:
            results: 模型结果，格式为 {benchmark: {metric: value}}
            baselines: 基线结果，格式为 {model_name: {benchmark: {metric: value}}}
            benchmarks: 基准测试列表
        
        Returns:
            LaTeX表格代码
        """
        if benchmarks is None:
            benchmarks = list(results.keys())
        
        # 确定所有指标
        all_metrics = set()
        for bench_results in results.values():
            all_metrics.update(bench_results.keys())
        
        metrics = sorted(all_metrics)
        
        # 构建表格
        lines = []
        lines.append("\\begin{table*}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Main Results Comparison with SOTA Methods}")
        lines.append("\\label{tab:main_results}")
        
        # 计算列数
        num_cols = 1 + len(benchmarks) * len(metrics)
        if baselines:
            num_cols += len(baselines) * len(metrics)
        
        col_spec = "l" + "c" * (num_cols - 1)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # 表头
        header = ["Method"]
        for bench in benchmarks:
            for metric in metrics:
                metric_name = metric.replace('_', ' ').title()
                header.append(f"{bench}\\\\{metric_name}")
        
        if baselines:
            for model_name in baselines.keys():
                for metric in metrics:
                    metric_name = metric.replace('_', ' ').title()
                    header.append(f"{model_name}\\\\{metric_name}")
        
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        
        # 我们的方法
        row = ["Ours"]
        for bench in benchmarks:
            bench_results = results.get(bench, {})
            for metric in metrics:
                value = bench_results.get(metric, None)
                row.append(self.format_number(value, bold=True))
        
        if baselines:
            for model_name in baselines.keys():
                model_results = baselines[model_name]
                for bench in benchmarks:
                    bench_results = model_results.get(bench, {})
                    for metric in metrics:
                        value = bench_results.get(metric, None)
                        row.append(self.format_number(value))
        
        lines.append(" & ".join(row) + " \\\\")
        
        # 基线方法
        if baselines:
            for model_name in baselines.keys():
                row = [self.escape_latex(model_name)]
                model_results = baselines[model_name]
                for bench in benchmarks:
                    bench_results = model_results.get(bench, {})
                    for metric in metrics:
                        value = bench_results.get(metric, None)
                        row.append(self.format_number(value))
                
                # 其他基线（跳过自己的列）
                for other_model in baselines.keys():
                    if other_model == model_name:
                        continue
                    other_results = baselines[other_model]
                    for bench in benchmarks:
                        bench_results = other_results.get(bench, {})
                        for metric in metrics:
                            value = bench_results.get(metric, None)
                            row.append(self.format_number(value))
                
                lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table*}")
        
        return "\n".join(lines)
    
    def generate_ablation_table(
        self,
        ablation_results: Dict[str, Dict[str, float]],
        baseline_result: Optional[Dict[str, float]] = None,
        metrics: Optional[List[str]] = None,
    ) -> str:
        """
        生成消融研究表
        
        Args:
            ablation_results: 消融实验结果，格式为 {experiment_name: {metric: value}}
            baseline_result: 基线结果（完整模型）
            metrics: 指标列表
        
        Returns:
            LaTeX表格代码
        """
        if metrics is None:
            metrics = set()
            for exp_results in ablation_results.values():
                metrics.update(exp_results.keys())
            metrics = sorted(metrics)
        
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Ablation Study Results}")
        lines.append("\\label{tab:ablation}")
        
        num_cols = 1 + len(metrics)
        col_spec = "l" + "c" * len(metrics)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # 表头
        header = ["Configuration"] + [m.replace('_', ' ').title() for m in metrics]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        
        # 基线（完整模型）
        if baseline_result:
            row = ["Full Model"]
            for metric in metrics:
                value = baseline_result.get(metric, None)
                row.append(self.format_number(value, bold=True))
            lines.append(" & ".join(row) + " \\\\")
            lines.append("\\midrule")
        
        # 消融实验
        for exp_name, exp_results in ablation_results.items():
            # 格式化实验名称
            display_name = exp_name.replace('_', ' ').title()
            if exp_name.startswith('wo_'):
                display_name = f"w/o {exp_name[3:].replace('_', ' ').title()}"
            
            row = [self.escape_latex(display_name)]
            for metric in metrics:
                value = exp_results.get(metric, None)
                row.append(self.format_number(value))
            
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_robustness_table(
        self,
        robustness_results: Dict[str, Dict[str, float]],
        benchmarks: Optional[List[str]] = None,
    ) -> str:
        """
        生成鲁棒性分析表
        
        Args:
            robustness_results: 鲁棒性结果，格式为 {benchmark: {metric: value}}
            benchmarks: 基准测试列表
        
        Returns:
            LaTeX表格代码
        """
        if benchmarks is None:
            benchmarks = list(robustness_results.keys())
        
        # 定义鲁棒性指标
        robustness_metrics = [
            'base_accuracy',
            'robust_accuracy',
            'selection_accuracy',
            'performance_gap',
            'temporal_sensitivity',
        ]
        
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Robustness Analysis}")
        lines.append("\\label{tab:robustness}")
        
        num_cols = 1 + len(robustness_metrics)
        col_spec = "l" + "c" * len(robustness_metrics)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # 表头
        metric_names = {
            'base_accuracy': 'Base Acc.',
            'robust_accuracy': 'Robust Acc.',
            'selection_accuracy': 'Selection Acc.',
            'performance_gap': 'Perf. Gap',
            'temporal_sensitivity': 'Temp. Sens.',
        }
        header = ["Benchmark"] + [metric_names.get(m, m.replace('_', ' ').title()) for m in robustness_metrics]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        
        # 数据行
        for bench in benchmarks:
            bench_results = robustness_results.get(bench, {})
            row = [self.escape_latex(bench)]
            for metric in robustness_metrics:
                value = bench_results.get(metric, None)
                if metric in ['base_accuracy', 'robust_accuracy', 'selection_accuracy']:
                    row.append(self.format_number(value * 100, unit="\\%"))
                else:
                    row.append(self.format_number(value))
            lines.append(" & ".join(row) + " \\\\")
        
        # 平均值行
        row = ["\\textbf{Average}"]
        for metric in robustness_metrics:
            values = [robustness_results.get(bench, {}).get(metric, 0) for bench in benchmarks]
            avg_value = np.mean([v for v in values if v is not None and not np.isnan(v)])
            if metric in ['base_accuracy', 'robust_accuracy', 'selection_accuracy']:
                row.append(self.format_number(avg_value * 100, unit="\\%", bold=True))
            else:
                row.append(self.format_number(avg_value, bold=True))
        lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_efficiency_table(
        self,
        efficiency_results: Dict[str, Dict[str, float]],
        models: Optional[List[str]] = None,
    ) -> str:
        """
        生成计算效率对比表
        
        Args:
            efficiency_results: 效率结果，格式为 {model: {metric: value}}
            models: 模型列表
        
        Returns:
            LaTeX表格代码
        """
        if models is None:
            models = list(efficiency_results.keys())
        
        # 定义效率指标
        efficiency_metrics = [
            'inference_time_ms',
            'memory_usage_mb',
            'throughput_samples_per_sec',
            'cache_hit_rate',
            'early_exit_rate',
        ]
        
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Computational Efficiency Comparison}")
        lines.append("\\label{tab:efficiency}")
        
        num_cols = 1 + len(efficiency_metrics)
        col_spec = "l" + "c" * len(efficiency_metrics)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # 表头
        metric_names = {
            'inference_time_ms': 'Time (ms)',
            'memory_usage_mb': 'Memory (MB)',
            'throughput_samples_per_sec': 'Throughput (s/s)',
            'cache_hit_rate': 'Cache Hit',
            'early_exit_rate': 'Early Exit',
        }
        header = ["Model"] + [metric_names.get(m, m.replace('_', ' ').title()) for m in efficiency_metrics]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        
        # 数据行
        for model in models:
            model_results = efficiency_results.get(model, {})
            row = [self.escape_latex(model)]
            for metric in efficiency_metrics:
                value = model_results.get(metric, None)
                if metric in ['cache_hit_rate', 'early_exit_rate']:
                    row.append(self.format_number(value * 100, unit="\\%"))
                elif metric == 'inference_time_ms':
                    row.append(self.format_number(value, unit=""))
                elif metric == 'memory_usage_mb':
                    row.append(self.format_number(value, unit=""))
                elif metric == 'throughput_samples_per_sec':
                    row.append(self.format_number(value, unit=""))
                else:
                    row.append(self.format_number(value))
            lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_selection_analysis_table(
        self,
        selection_results: Dict[str, Dict[str, float]],
        benchmarks: Optional[List[str]] = None,
    ) -> str:
        """
        生成选择决策分析表
        
        Args:
            selection_results: 选择结果，格式为 {benchmark: {metric: value}}
            benchmarks: 基准测试列表
        
        Returns:
            LaTeX表格代码
        """
        if benchmarks is None:
            benchmarks = list(selection_results.keys())
        
        # 定义选择指标
        selection_metrics = [
            'clean_selection_rate',
            'noisy_selection_rate',
            'correct_selection_rate',
            'avg_selection_confidence',
            'early_exit_rate',
        ]
        
        lines = []
        lines.append("\\begin{table}[t]")
        lines.append("\\centering")
        lines.append("\\caption{Selection Decision Analysis}")
        lines.append("\\label{tab:selection}")
        
        num_cols = 1 + len(selection_metrics)
        col_spec = "l" + "c" * len(selection_metrics)
        lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
        lines.append("\\toprule")
        
        # 表头
        metric_names = {
            'clean_selection_rate': 'Clean Sel.',
            'noisy_selection_rate': 'Noisy Sel.',
            'correct_selection_rate': 'Correct Sel.',
            'avg_selection_confidence': 'Avg Conf.',
            'early_exit_rate': 'Early Exit',
        }
        header = ["Benchmark"] + [metric_names.get(m, m.replace('_', ' ').title()) for m in selection_metrics]
        lines.append(" & ".join(header) + " \\\\")
        lines.append("\\midrule")
        
        # 数据行
        for bench in benchmarks:
            bench_results = selection_results.get(bench, {})
            row = [self.escape_latex(bench)]
            for metric in selection_metrics:
                value = bench_results.get(metric, None)
                if metric in ['clean_selection_rate', 'noisy_selection_rate', 'correct_selection_rate', 'early_exit_rate']:
                    row.append(self.format_number(value * 100, unit="\\%"))
                elif metric == 'avg_selection_confidence':
                    row.append(self.format_number(value))
                else:
                    row.append(self.format_number(value))
            lines.append(" & ".join(row) + " \\\\")
        
        # 平均值行
        row = ["\\textbf{Average}"]
        for metric in selection_metrics:
            values = [selection_results.get(bench, {}).get(metric, 0) for bench in benchmarks]
            avg_value = np.mean([v for v in values if v is not None and not np.isnan(v)])
            if metric in ['clean_selection_rate', 'noisy_selection_rate', 'correct_selection_rate', 'early_exit_rate']:
                row.append(self.format_number(avg_value * 100, unit="\\%", bold=True))
            else:
                row.append(self.format_number(avg_value, bold=True))
        lines.append(" & ".join(row) + " \\\\")
        
        lines.append("\\bottomrule")
        lines.append("\\end{tabular}")
        lines.append("\\end{table}")
        
        return "\n".join(lines)
    
    def generate_all_tables(
        self,
        results_file: str,
        output_dir: str = "./latex_tables",
    ) -> Dict[str, str]:
        """
        生成所有表格
        
        Args:
            results_file: 结果文件路径（JSON格式）
            output_dir: 输出目录
        
        Returns:
            字典，键为表格类型，值为LaTeX代码
        """
        # 加载结果
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = {}
        
        # 1. 主结果表
        if 'main_results' in data:
            main_results = data['main_results']
            baselines = data.get('baselines', None)
            tables['main_results'] = self.generate_main_results_table(
                main_results, baselines
            )
        
        # 2. 消融研究表
        if 'ablation_results' in data:
            ablation_results = data['ablation_results']
            baseline = data.get('baseline_result', None)
            tables['ablation'] = self.generate_ablation_table(
                ablation_results, baseline
            )
        
        # 3. 鲁棒性分析表
        if 'robustness_results' in data:
            robustness_results = data['robustness_results']
            tables['robustness'] = self.generate_robustness_table(
                robustness_results
            )
        
        # 4. 计算效率表
        if 'efficiency_results' in data:
            efficiency_results = data['efficiency_results']
            tables['efficiency'] = self.generate_efficiency_table(
                efficiency_results
            )
        
        # 5. 选择决策分析表
        if 'selection_results' in data:
            selection_results = data['selection_results']
            tables['selection'] = self.generate_selection_analysis_table(
                selection_results
            )
        
        # 保存表格
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        for table_type, latex_code in tables.items():
            output_path = Path(output_dir) / f"{table_type}_table.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"Generated {table_type} table: {output_path}")
        
        # 生成汇总文件
        summary_path = Path(output_dir) / "all_tables.tex"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("% All LaTeX Tables\n")
            f.write("% Generated automatically\n\n")
            for table_type, latex_code in tables.items():
                f.write(f"% {table_type.upper()} TABLE\n")
                f.write(latex_code)
                f.write("\n\n")
        print(f"Generated summary file: {summary_path}")
        
        return tables


def main():
    parser = argparse.ArgumentParser(description="LaTeX Table Generator")
    parser.add_argument(
        '--results-file',
        type=str,
        required=True,
        help='Path to results JSON file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./latex_tables',
        help='Output directory for LaTeX files'
    )
    parser.add_argument(
        '--table-type',
        type=str,
        choices=['main', 'ablation', 'robustness', 'efficiency', 'selection', 'all'],
        default='all',
        help='Type of table to generate'
    )
    parser.add_argument(
        '--decimal-places',
        type=int,
        default=2,
        help='Number of decimal places'
    )
    
    args = parser.parse_args()
    
    generator = LaTeXTableGenerator(decimal_places=args.decimal_places)
    
    if args.table_type == 'all':
        tables = generator.generate_all_tables(args.results_file, args.output_dir)
        print(f"\nGenerated {len(tables)} tables")
    else:
        # 加载结果
        with open(args.results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
        if args.table_type == 'main':
            main_results = data.get('main_results', {})
            baselines = data.get('baselines', None)
            latex_code = generator.generate_main_results_table(main_results, baselines)
            output_path = Path(args.output_dir) / "main_results_table.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"Generated main results table: {output_path}")
        
        elif args.table_type == 'ablation':
            ablation_results = data.get('ablation_results', {})
            baseline = data.get('baseline_result', None)
            latex_code = generator.generate_ablation_table(ablation_results, baseline)
            output_path = Path(args.output_dir) / "ablation_table.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"Generated ablation table: {output_path}")
        
        elif args.table_type == 'robustness':
            robustness_results = data.get('robustness_results', {})
            latex_code = generator.generate_robustness_table(robustness_results)
            output_path = Path(args.output_dir) / "robustness_table.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"Generated robustness table: {output_path}")
        
        elif args.table_type == 'efficiency':
            efficiency_results = data.get('efficiency_results', {})
            latex_code = generator.generate_efficiency_table(efficiency_results)
            output_path = Path(args.output_dir) / "efficiency_table.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"Generated efficiency table: {output_path}")
        
        elif args.table_type == 'selection':
            selection_results = data.get('selection_results', {})
            latex_code = generator.generate_selection_analysis_table(selection_results)
            output_path = Path(args.output_dir) / "selection_table.tex"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(latex_code)
            print(f"Generated selection analysis table: {output_path}")


if __name__ == "__main__":
    main()

