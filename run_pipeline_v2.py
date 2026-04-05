"""
完整流程运行脚本 V2 - 分范式编码架构
步骤1: 数据预处理 (AU → GASF, 每个范式独立存储)
步骤2: 监督对比学习预训练 (分范式编码+融合)
步骤3: 有监督微调 (冻结 Encoder)
"""
import os
import sys
from preprocess_asd2_v2 import preprocess
from stage1_pretrain_v2 import pretrain
from stage2_finetune_v2 import finetune


def main():
    print("="*70)
    print(" " * 10 + "🧠 ASD 筛查完整流程 V2 (分范式编码) 🧠")
    print("="*70)
    print("\n架构流程:")
    print("  1️⃣  AU 数据 → GASF 编码 (每个范式独立: A1, A2, C, D)")
    print("  2️⃣  构造正负样本对 (同类别=正样本, 异类别=负样本)")
    print("  3️⃣  监督对比学习预训练:")
    print("      - 每个范式独立编码 (17通道 → 256维)")
    print("      - 共享Encoder权重")
    print("      - 融合4个范式特征 (1024维 → 256维)")
    print("  4️⃣  冻结 Encoder → 有监督微调")
    print("  5️⃣  ASD/TD 二分类评估 (10-Fold CV)")
    print("="*70)

    # ============ 步骤1: 数据预处理 ============
    print("\n" + "🔄 "*15)
    print("步骤1: 数据预处理 (AU → GASF 编码)")
    print("🔄 "*15)

    if not os.path.exists("AU-ASD-TD-GASF-V2"):
        print("⚠️  未找到预处理数据，开始执行预处理...")
        preprocess()
    else:
        user_input = input("⚠️  检测到已有 GASF 数据，是否重新预处理？ (y/n): ").strip().lower()
        if user_input == 'y':
            preprocess()
        else:
            print("✅ 跳过预处理步骤，使用现有数据。")

    # ============ 步骤2: 对比学习预训练 ============
    print("\n" + "🔄 "*15)
    print("步骤2: 监督对比学习预训练 (分范式编码)")
    print("🔄 "*15)

    pretrain_config = {
        'processed_root': 'TDA-GASF',
        'save_path': 'pretrained_encoder_TDA.pth',
        'batch_size': 8,
        'epochs': 50,
        'lr': 1e-3,
        'temperature': 0.5,
        'fusion_type': 'concat',  # 可选: 'concat', 'mean', 'attention'
        'seed': 42
    }

    if not os.path.exists(pretrain_config['save_path']):
        print("⚠️  未找到预训练模型，开始训练...")
        pretrain(**pretrain_config)
    else:
        user_input = input("⚠️  检测到已有预训练模型，是否重新训练？ (y/n): ").strip().lower()
        if user_input == 'y':
            pretrain(**pretrain_config)
        else:
            print("✅ 跳过预训练步骤，使用现有模型。")

    # ============ 步骤3: 有监督微调 ============
    print("\n" + "🔄 "*15)
    print("步骤3: 有监督微调 (冻结 Encoder)")
    print("🔄 "*15)

    finetune_config = {
        'processed_root': 'TDA-GASF',
        'pretrained_encoder_path': 'pretrained_encoder_TDA.pth',
        'batch_size': 8,
        'epochs': 30,
        'lr': 1e-3,
        'fusion_type': 'concat',
        'seed': 42
    }

    acc, sen, spe, auc = finetune(**finetune_config)

    # ============ 完成 ============
    print("\n\n" + "🎉"*20)
    print(" " * 20 + "流程执行完成！")
    print("🎉"*20)
    print(f"\n最终结果:")
    print(f"  📊 Accuracy   : {acc:.4f}")
    print(f"  📊 Sensitivity: {sen:.4f}")
    print(f"  📊 Specificity: {spe:.4f}")
    print(f"  📊 AUC Score  : {auc:.4f}")
    print("="*70)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断执行。")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
