"""
代码逻辑测试脚本 V2 - 分范式编码架构
测试数据加载、模型前向传播、损失计算等
"""
import torch
import os
from contrastive_model_v2 import MultiTaskEncoder, SingleTaskEncoder, ClassificationHead, ContrastiveModelV2
from stage1_pretrain_v2 import SupervisedContrastiveLoss


def test_single_task_encoder():
    """测试单范式编码器"""
    print("\n" + "="*60)
    print("测试1: 单范式编码器")
    print("="*60)

    device = torch.device("cpu")
    batch_size = 4

    try:
        encoder = SingleTaskEncoder(in_channels=17, feature_dim=256).to(device)
        dummy_input = torch.randn(batch_size, 17, 64, 64).to(device)

        features = encoder(dummy_input)
        print(f"✅ 单范式编码器测试成功:")
        print(f"   - 输入: {dummy_input.shape}")
        print(f"   - 输出: {features.shape}")

        assert features.shape == (batch_size, 256), "输出维度错误！"
        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_task_encoder():
    """测试多范式编码器"""
    print("\n" + "="*60)
    print("测试2: 多范式编码器（分范式编码+融合）")
    print("="*60)

    device = torch.device("cpu")
    batch_size = 4

    try:
        # 测试不同融合策略
        for fusion_type in ['concat', 'mean', 'attention']:
            print(f"\n测试融合策略: {fusion_type}")
            encoder = MultiTaskEncoder(num_tasks=4, feature_dim=256, fusion_type=fusion_type).to(device)

            # 模拟4个范式的输入
            task_dict = {
                'A1': torch.randn(batch_size, 17, 64, 64).to(device),
                'A2': torch.randn(batch_size, 17, 64, 64).to(device),
                'C': torch.randn(batch_size, 17, 64, 64).to(device),
                'D': torch.randn(batch_size, 17, 64, 64).to(device)
            }

            fused_features = encoder(task_dict)
            print(f"   ✅ {fusion_type} 融合成功: {fused_features.shape}")

            assert fused_features.shape == (batch_size, 256), f"{fusion_type} 输出维度错误！"

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_contrastive_model():
    """测试完整对比学习模型"""
    print("\n" + "="*60)
    print("测试3: 完整对比学习模型")
    print("="*60)

    device = torch.device("cpu")
    batch_size = 4

    try:
        model = ContrastiveModelV2(encoder_dim=256, projection_dim=64, fusion_type='concat').to(device)

        task_dict = {
            'A1': torch.randn(batch_size, 17, 64, 64).to(device),
            'A2': torch.randn(batch_size, 17, 64, 64).to(device),
            'C': torch.randn(batch_size, 17, 64, 64).to(device),
            'D': torch.randn(batch_size, 17, 64, 64).to(device)
        }

        features, projections = model(task_dict)
        print(f"✅ 完整模型测试成功:")
        print(f"   - Features: {features.shape}")
        print(f"   - Projections: {projections.shape}")

        assert features.shape == (batch_size, 256), "Features 维度错误！"
        assert projections.shape == (batch_size, 64), "Projections 维度错误！"

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation():
    """测试损失计算"""
    print("\n" + "="*60)
    print("测试4: Supervised Contrastive Loss")
    print("="*60)

    device = torch.device("cpu")
    batch_size = 4

    try:
        # 创建模拟数据
        anchor = torch.randn(batch_size, 64).to(device)
        positive = torch.randn(batch_size, 64).to(device)
        negative = torch.randn(batch_size, 64).to(device)

        # 归一化
        anchor = torch.nn.functional.normalize(anchor, dim=1)
        positive = torch.nn.functional.normalize(positive, dim=1)
        negative = torch.nn.functional.normalize(negative, dim=1)

        # 计算损失
        criterion = SupervisedContrastiveLoss(temperature=0.5)
        loss = criterion(anchor, positive, negative)

        print(f"✅ 损失计算成功:")
        print(f"   - Loss value: {loss.item():.4f}")

        assert not torch.isnan(loss), "Loss 是 NaN！"
        assert not torch.isinf(loss), "Loss 是 Inf！"

        return True
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structure():
    """测试真实数据结构"""
    print("\n" + "="*60)
    print("测试5: 真实数据加载（如果数据已预处理）")
    print("="*60)

    processed_root = "AU-ASD-TD-GASF-V2"

    if not os.path.exists(processed_root):
        print(f"⚠️  预处理数据不存在: {processed_root}")
        print("   请先运行: python preprocess_asd2_v2.py")
        return None

    try:
        # 尝试加载一个文件
        for group in ['ASD', 'TD']:
            group_dir = os.path.join(processed_root, group)
            if os.path.exists(group_dir):
                files = [f for f in os.listdir(group_dir) if f.endswith('.pt')]
                if len(files) > 0:
                    sample_file = os.path.join(group_dir, files[0])
                    data = torch.load(sample_file, weights_only=True)

                    print(f"✅ 数据加载测试成功:")
                    print(f"   - 文件: {sample_file}")
                    print(f"   - 数据类型: {type(data)}")
                    print(f"   - 包含范式: {list(data.keys())}")
                    for task, tensor in data.items():
                        print(f"   - {task}: {tensor.shape}")

                    # 验证数据结构
                    assert isinstance(data, dict), "数据应该是字典格式！"
                    assert all(k in data for k in ['A1', 'A2', 'C', 'D']), "缺少某些范式！"
                    assert all(v.shape == (17, 64, 64) for v in data.values()), "范式数据维度错误！"

                    return True

        print("⚠️  未找到有效数据文件")
        return None

    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "🧪"*30)
    print(" "*20 + "代码逻辑测试 V2")
    print("🧪"*30)

    results = []

    # 运行测试
    results.append(("单范式编码器", test_single_task_encoder()))
    results.append(("多范式编码器", test_multi_task_encoder()))
    results.append(("完整对比模型", test_contrastive_model()))
    results.append(("损失函数", test_loss_computation()))

    # 数据结构测试（可选）
    data_test = test_data_structure()
    if data_test is not None:
        results.append(("数据加载", data_test))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    all_passed = all(r[1] for r in results if r[1] is not None)
    if all_passed:
        print("\n🎉 所有测试通过！代码逻辑正确。")
        print("\n下一步:")
        print("  1. 运行预处理: python preprocess_asd2_v2.py")
        print("  2. 运行完整流程: python run_pipeline_v2.py")
    else:
        print("\n⚠️  部分测试失败，请检查错误信息。")

    return all_passed


if __name__ == '__main__':
    main()
