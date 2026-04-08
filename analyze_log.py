"""
训练日志分析工具
用于查看和分析训练日志中的关键信息
"""

import pandas as pd
import os


def analyze_training_log(log_file='logs/training_log.csv'):
    """
    分析训练日志并打印关键统计信息
    """
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        return
    
    df = pd.read_csv(log_file)
    
    if len(df) == 0:
        print("❌ 日志文件为空")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 训练日志分析报告")
    print(f"{'='*70}")
    print(f"日志文件: {log_file}")
    print(f"数据条数: {len(df)}")
    print(f"{'='*70}\n")
    
    # 1. 基础统计
    print(f"📌 基础信息:")
    print(f"   训练轮次: {df['epoch'].min()} ~ {df['epoch'].max()}")
    print(f"   总训练时间: {df['epoch_time'].sum() / 3600:.2f} 小时")
    print(f"   平均每轮时间: {df['epoch_time'].mean():.1f} 秒")
    print(f"   最快一轮: {df['epoch_time'].min():.1f} 秒 (Epoch {df.loc[df['epoch_time'].idxmin(), 'epoch']})")
    print(f"   最慢一轮: {df['epoch_time'].max():.1f} 秒 (Epoch {df.loc[df['epoch_time'].idxmax(), 'epoch']})")
    
    # 2. Loss统计
    print(f"\n📉 损失统计:")
    print(f"   初始Loss: {df['train_loss_total'].iloc[0]:.4f}")
    print(f"   最终Loss: {df['train_loss_total'].iloc[-1]:.4f}")
    print(f"   最小Loss: {df['train_loss_total'].min():.4f} (Epoch {df.loc[df['train_loss_total'].idxmin(), 'epoch']})")
    print(f"   Loss下降: {df['train_loss_total'].iloc[0] - df['train_loss_total'].iloc[-1]:.4f} "
          f"({(df['train_loss_total'].iloc[0] - df['train_loss_total'].iloc[-1]) / df['train_loss_total'].iloc[0] * 100:.1f}%)")
    
    print(f"\n   各分项最终Loss:")
    print(f"     - 分类Loss: {df['train_loss_cls'].iloc[-1]:.4f}")
    print(f"     - 回归Loss: {df['train_loss_box'].iloc[-1]:.4f}")
    print(f"     - DFL Loss: {df['train_loss_dfl'].iloc[-1]:.4f}")
    
    # 3. 学习率统计
    print(f"\n📈 学习率变化:")
    print(f"   初始学习率: {df['learning_rate'].iloc[0]:.6f}")
    print(f"   最终学习率: {df['learning_rate'].iloc[-1]:.6f}")
    print(f"   最大学习率: {df['learning_rate'].max():.6f}")
    print(f"   最小学习率: {df['learning_rate'].min():.6f}")
    
    # 4. 验证指标统计
    val_df = df[df['val_mAP50'] > 0].copy()
    
    if len(val_df) > 0:
        print(f"\n🎯 验证指标统计:")
        print(f"   验证次数: {len(val_df)}")
        
        best_idx = val_df['val_mAP50'].idxmax()
        print(f"\n   最佳结果 (Epoch {int(val_df.loc[best_idx, 'epoch'])}):")
        print(f"     - mAP@50: {val_df.loc[best_idx, 'val_mAP50']:.4f}")
        print(f"     - Precision: {val_df.loc[best_idx, 'val_precision']:.4f}")
        print(f"     - Recall: {val_df.loc[best_idx, 'val_recall']:.4f}")
        
        print(f"\n   平均指标:")
        print(f"     - 平均mAP@50: {val_df['val_mAP50'].mean():.4f}")
        print(f"     - 平均Precision: {val_df['val_precision'].mean():.4f}")
        print(f"     - 平均Recall: {val_df['val_recall'].mean():.4f}")
        
        print(f"\n   指标范围:")
        print(f"     - mAP@50: {val_df['val_mAP50'].min():.4f} ~ {val_df['val_mAP50'].max():.4f}")
        print(f"     - Precision: {val_df['val_precision'].min():.4f} ~ {val_df['val_precision'].max():.4f}")
        print(f"     - Recall: {val_df['val_recall'].min():.4f} ~ {val_df['val_recall'].max():.4f}")
    else:
        print(f"\n⚠️ 暂无验证数据")
    
    # 5. 训练趋势分析
    print(f"\n📊 训练趋势分析:")
    
    # 最近10轮的平均Loss
    recent_n = min(10, len(df))
    recent_loss = df['train_loss_total'].iloc[-recent_n:].mean()
    print(f"   最近{recent_n}轮平均Loss: {recent_loss:.4f}")
    
    # Loss下降趋势
    if len(df) >= 10:
        first_10_avg = df['train_loss_total'].iloc[:10].mean()
        last_10_avg = df['train_loss_total'].iloc[-10:].mean()
        improvement = (first_10_avg - last_10_avg) / first_10_avg * 100
        print(f"   前10轮vs后10轮Loss改善: {improvement:.1f}%")
    
    # 6. 最近5轮详情
    print(f"\n📋 最近5轮训练详情:")
    print(f"{'Epoch':<8} {'Total Loss':<12} {'Cls Loss':<12} {'Box Loss':<12} {'DFL Loss':<12} {'LR':<12} {'Time':<8}")
    print("-" * 80)
    for idx in range(max(0, len(df)-5), len(df)):
        row = df.iloc[idx]
        print(f"{int(row['epoch']):<8} {row['train_loss_total']:<12.4f} {row['train_loss_cls']:<12.4f} "
              f"{row['train_loss_box']:<12.4f} {row['train_loss_dfl']:<12.4f} {row['learning_rate']:<12.6f} "
              f"{row['epoch_time']:<8.1f}s")
    
    print(f"\n{'='*70}\n")


def export_summary(log_file='logs/training_log.csv', output_file='logs/training_summary.txt'):
    """
    导出训练摘要到文本文件
    """
    import sys
    from io import StringIO
    
    # 重定向输出
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    analyze_training_log(log_file)
    
    # 获取输出内容
    output = sys.stdout.getvalue()
    
    # 恢复标准输出
    sys.stdout = old_stdout
    
    # 保存到文件
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(output)
    
    print(f"✅ 训练摘要已导出至: {output_file}")
    print(output)


if __name__ == "__main__":
    analyze_training_log()
