"""
训练日志可视化工具
用于绘制训练过程中的Loss、mAP等指标曲线
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def plot_training_curves(log_file='logs/training_log.csv', output_dir='logs/figures'):
    """
    绘制训练曲线
    
    参数:
        log_file: 训练日志CSV文件路径
        output_dir: 图片保存目录
    """
    if not os.path.exists(log_file):
        print(f"❌ 日志文件不存在: {log_file}")
        print("   请先运行训练脚本生成日志文件")
        return
    
    df = pd.read_csv(log_file)
    
    if len(df) == 0:
        print("❌ 日志文件为空")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n📊 正在生成训练曲线...")
    print(f"   日志文件: {log_file}")
    print(f"   数据条数: {len(df)}")
    print(f"   保存目录: {output_dir}\n")
    
    # 1. 绘制Loss曲线
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练损失曲线', fontsize=16, fontweight='bold')
    
    # 总Loss
    axes[0, 0].plot(df['epoch'], df['train_loss_total'], 'b-', linewidth=2, label='Total Loss')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('总损失 (Total Loss)', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=10)
    
    # 分类Loss
    axes[0, 1].plot(df['epoch'], df['train_loss_cls'], 'r-', linewidth=2, label='Classification Loss')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('分类损失 (Classification Loss)', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # 回归Loss
    axes[1, 0].plot(df['epoch'], df['train_loss_box'], 'g-', linewidth=2, label='Box Loss')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Loss', fontsize=12)
    axes[1, 0].set_title('回归损失 (Box Loss)', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=10)
    
    # DFL Loss
    axes[1, 1].plot(df['epoch'], df['train_loss_dfl'], 'm-', linewidth=2, label='DFL Loss')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('Loss', fontsize=12)
    axes[1, 1].set_title('分布焦点损失 (DFL Loss)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    loss_fig_path = os.path.join(output_dir, 'loss_curves.png')
    plt.savefig(loss_fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ Loss曲线已保存: {loss_fig_path}")
    plt.close()
    
    # 2. 绘制验证指标曲线
    val_df = df[df['val_mAP50'] > 0].copy()
    
    if len(val_df) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('验证指标曲线', fontsize=16, fontweight='bold')
        
        # mAP50
        axes[0].plot(val_df['epoch'], val_df['val_mAP50'], 'b-o', linewidth=2, markersize=6, label='mAP@50')
        axes[0].axhline(y=val_df['val_mAP50'].max(), color='r', linestyle='--', label=f'Best: {val_df["val_mAP50"].max():.4f}')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('mAP@50', fontsize=12)
        axes[0].set_title('平均精度 (mAP@50)', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(fontsize=10)
        
        # Precision
        axes[1].plot(val_df['epoch'], val_df['val_precision'], 'g-o', linewidth=2, markersize=6, label='Precision')
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Precision', fontsize=12)
        axes[1].set_title('精确率 (Precision)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(fontsize=10)
        
        # Recall
        axes[2].plot(val_df['epoch'], val_df['val_recall'], 'm-o', linewidth=2, markersize=6, label='Recall')
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('Recall', fontsize=12)
        axes[2].set_title('召回率 (Recall)', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(fontsize=10)
        
        plt.tight_layout()
        metrics_fig_path = os.path.join(output_dir, 'metrics_curves.png')
        plt.savefig(metrics_fig_path, dpi=300, bbox_inches='tight')
        print(f"✅ 验证指标曲线已保存: {metrics_fig_path}")
        plt.close()
    else:
        print("⚠️ 暂无验证数据，跳过验证指标曲线绘制")
    
    # 3. 绘制学习率曲线
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(df['epoch'], df['learning_rate'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('学习率变化曲线', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    lr_fig_path = os.path.join(output_dir, 'learning_rate.png')
    plt.savefig(lr_fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ 学习率曲线已保存: {lr_fig_path}")
    plt.close()
    
    # 4. 绘制综合对比图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('训练过程综合分析', fontsize=16, fontweight='bold')
    
    # Loss + mAP
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'], df['train_loss_total'], 'b-', linewidth=2, label='Total Loss')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True, alpha=0.3)
    
    if len(val_df) > 0:
        ax2 = ax1.twinx()
        ax2.plot(val_df['epoch'], val_df['val_mAP50'], 'r-o', linewidth=2, markersize=4, label='mAP@50')
        ax2.set_ylabel('mAP@50', fontsize=12, color='r')
        ax2.tick_params(axis='y', labelcolor='r')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    if len(val_df) > 0:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    else:
        ax1.legend(fontsize=10)
    ax1.set_title('Loss vs mAP', fontsize=14, fontweight='bold')
    
    # 所有Loss对比
    axes[0, 1].plot(df['epoch'], df['train_loss_total'], 'b-', linewidth=2, label='Total', alpha=0.8)
    axes[0, 1].plot(df['epoch'], df['train_loss_cls'], 'r-', linewidth=1.5, label='Cls', alpha=0.7)
    axes[0, 1].plot(df['epoch'], df['train_loss_box'], 'g-', linewidth=1.5, label='Box', alpha=0.7)
    axes[0, 1].plot(df['epoch'], df['train_loss_dfl'], 'm-', linewidth=1.5, label='DFL', alpha=0.7)
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].set_title('所有损失对比', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=10)
    
    # Precision vs Recall
    if len(val_df) > 0:
        axes[1, 0].plot(val_df['val_recall'], val_df['val_precision'], 'go-', linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Recall', fontsize=12)
        axes[1, 0].set_ylabel('Precision', fontsize=12)
        axes[1, 0].set_title('Precision-Recall 曲线', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 标注最佳点
        best_idx = val_df['val_mAP50'].idxmax()
        best_epoch = val_df.loc[best_idx, 'epoch']
        best_precision = val_df.loc[best_idx, 'val_precision']
        best_recall = val_df.loc[best_idx, 'val_recall']
        axes[1, 0].scatter([best_recall], [best_precision], color='red', s=200, zorder=5, marker='*')
        axes[1, 0].annotate(f'Best (Epoch {int(best_epoch)})', 
                           xy=(best_recall, best_precision),
                           xytext=(10, 10), textcoords='offset points',
                           fontsize=10, color='red',
                           arrowprops=dict(arrowstyle='->', color='red'))
    else:
        axes[1, 0].text(0.5, 0.5, '暂无验证数据', ha='center', va='center', fontsize=14)
        axes[1, 0].set_title('Precision-Recall 曲线', fontsize=14, fontweight='bold')
    
    # 训练时间统计
    axes[1, 1].bar(df['epoch'], df['epoch_time'], color='steelblue', alpha=0.7)
    axes[1, 1].axhline(y=df['epoch_time'].mean(), color='r', linestyle='--', 
                       label=f'平均: {df["epoch_time"].mean():.1f}s')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('时间 (秒)', fontsize=12)
    axes[1, 1].set_title('每轮训练耗时', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].legend(fontsize=10)
    
    plt.tight_layout()
    summary_fig_path = os.path.join(output_dir, 'training_summary.png')
    plt.savefig(summary_fig_path, dpi=300, bbox_inches='tight')
    print(f"✅ 综合分析图已保存: {summary_fig_path}")
    plt.close()
    
    # 打印统计信息
    print(f"\n{'='*60}")
    print(f"📊 训练统计信息:")
    print(f"{'='*60}")
    print(f"总轮次: {len(df)}")
    print(f"总训练时间: {df['epoch_time'].sum() / 3600:.2f} 小时")
    print(f"平均每轮时间: {df['epoch_time'].mean():.1f} 秒")
    print(f"\n最终Loss:")
    print(f"  总Loss: {df['train_loss_total'].iloc[-1]:.4f}")
    print(f"  分类Loss: {df['train_loss_cls'].iloc[-1]:.4f}")
    print(f"  回归Loss: {df['train_loss_box'].iloc[-1]:.4f}")
    print(f"  DFL Loss: {df['train_loss_dfl'].iloc[-1]:.4f}")
    
    if len(val_df) > 0:
        print(f"\n最佳验证结果:")
        best_idx = val_df['val_mAP50'].idxmax()
        print(f"  Epoch: {int(val_df.loc[best_idx, 'epoch'])}")
        print(f"  mAP@50: {val_df.loc[best_idx, 'val_mAP50']:.4f}")
        print(f"  Precision: {val_df.loc[best_idx, 'val_precision']:.4f}")
        print(f"  Recall: {val_df.loc[best_idx, 'val_recall']:.4f}")
    
    print(f"{'='*60}\n")


if __name__ == "__main__":
    plot_training_curves()
