#!/usr/bin/env python3
"""
推送本地 wandb 数据到 wandb 服务器

用法:
    python push_wandb.py --wandb_key YOUR_WANDB_KEY --wandb_dir /path/to/wandb/folder
    python push_wandb.py -k YOUR_WANDB_KEY -d /path/to/wandb/folder
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def is_run_directory(path: Path) -> bool:
    """检查路径是否是一个 wandb run 目录"""
    if not path.is_dir():
        return False
    # 检查是否包含 wandb run 的特征文件/目录
    return (
        any(path.glob("run-*.wandb")) or  # 包含 .wandb 文件
        (path / "files").exists() or      # 包含 files 目录
        (path / "logs").exists()          # 包含 logs 目录
    )


def find_run_directories(wandb_dir: Path):
    """
    查找所有需要同步的 run 目录
    
    支持以下情况:
    1. 单个 run 目录 (如 run-20251219_094156-4ls2y2s3)
    2. 包含多个 run 目录的文件夹 (如 wandb/ 包含多个 run-xxx)
    3. 包含 runs 子目录的文件夹 (如 wandb/runs/)
    """
    run_dirs = []
    
    # 情况1: 如果本身就是 run 目录
    if is_run_directory(wandb_dir):
        return [wandb_dir]
    
    # 情况2: 检查是否有 runs 子目录
    runs_dir = wandb_dir / "runs"
    if runs_dir.exists() and runs_dir.is_dir():
        # 遍历 runs 目录下的所有子目录
        run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and is_run_directory(d)]
        if run_dirs:
            return run_dirs
    
    # 情况3: 直接在当前目录查找所有 run-* 目录
    run_dirs = [d for d in wandb_dir.iterdir() 
                if d.is_dir() and d.name.startswith("run-") and is_run_directory(d)]
    
    return run_dirs


def push_wandb_data(wandb_key: str, wandb_dir: str):
    """
    推送本地 wandb 数据到服务器
    
    Args:
        wandb_key: WandB API key
        wandb_dir: 本地 wandb 文件夹路径（可以是单个 run 目录、包含多个 run 的文件夹、或包含 runs 子目录的文件夹）
    """
    wandb_dir = Path(wandb_dir).expanduser().resolve()
    
    if not wandb_dir.exists():
        print(f"❌ 错误: wandb 文件夹不存在: {wandb_dir}")
        sys.exit(1)
    
    print(f"📁 搜索目录: {wandb_dir}")
    
    # 查找所有 run 目录
    run_dirs = find_run_directories(wandb_dir)
    
    if not run_dirs:
        print(f"⚠️  警告: 在 {wandb_dir} 中没有找到任何 run 目录")
        print(f"   提示: 请确保路径指向:")
        print(f"     - 单个 run 目录 (如 run-20251219_094156-4ls2y2s3)")
        print(f"     - 包含多个 run 目录的文件夹 (如 wandb/)")
        print(f"     - 包含 runs 子目录的文件夹")
        sys.exit(1)
    
    print(f"📊 找到 {len(run_dirs)} 个 run 目录")
    print()
    
    # 设置环境变量
    env = os.environ.copy()
    env["WANDB_API_KEY"] = wandb_key
    
    # 对每个 run 目录执行 wandb sync
    success_count = 0
    fail_count = 0
    
    for i, run_dir in enumerate(run_dirs, 1):
        print(f"[{i}/{len(run_dirs)}] 正在同步: {run_dir.name}")
        
        try:
            # 使用 wandb sync 命令同步
            result = subprocess.run(
                ["wandb", "sync", str(run_dir)],
                env=env,
                capture_output=True,
                text=True,
                timeout=300  # 5分钟超时
            )
            
            if result.returncode == 0:
                print(f"  ✅ 成功同步: {run_dir.name}")
                success_count += 1
            else:
                print(f"  ❌ 同步失败: {run_dir.name}")
                print(f"     错误信息: {result.stderr}")
                fail_count += 1
                
        except subprocess.TimeoutExpired:
            print(f"  ⏱️  超时: {run_dir.name} (超过5分钟)")
            fail_count += 1
        except Exception as e:
            print(f"  ❌ 异常: {run_dir.name}")
            print(f"     错误: {str(e)}")
            fail_count += 1
        
        print()
    
    # 总结
    print("=" * 60)
    print(f"📊 同步完成:")
    print(f"   ✅ 成功: {success_count}")
    print(f"   ❌ 失败: {fail_count}")
    print(f"   📁 总计: {len(run_dirs)}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="推送本地 wandb 数据到 wandb 服务器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 同步单个 run
  python push_wandb.py -k YOUR_KEY -d /root/data/code/verl-agent/wandb/run-20251219_094156-4ls2y2s3
  
  # 同步整个 wandb 目录下的所有 run
  python push_wandb.py -k YOUR_KEY -d /root/data/code/verl-agent/wandb
  
  # 同步 latest-run (符号链接)
  python push_wandb.py -k YOUR_KEY -d /root/data/code/verl-agent/wandb/latest-run
        """
    )
    
    parser.add_argument(
        "-k", "--wandb_key",
        required=False,
        default="354e1d0ee17771243321187ec0d3ba7bcc1105d0",
        help="WandB API key"
    )
    
    parser.add_argument(
        "-d", "--wandb_dir",
        required=False,
        default="/inspire/hdd/project/ai4education/qianhong-p-qianhong/verl-agent-proagent/verl-agent/wandb/offline-run-20251220_053816-wx5lq9ni",
        help="本地 wandb 路径（可以是单个 run 目录、包含多个 run 的文件夹、或包含 runs 子目录的文件夹）"
    )
    
    args = parser.parse_args()
    
    # 检查 wandb 命令是否可用
    try:
        subprocess.run(["wandb", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ 错误: wandb 命令未找到")
        print("   请先安装: pip install wandb")
        sys.exit(1)
    
    push_wandb_data(args.wandb_key, args.wandb_dir)


if __name__ == "__main__":
    main()

