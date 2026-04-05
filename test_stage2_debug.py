"""调试stage2微调问题"""
import torch
import os
import sys

print("1. 开始调试...")
print(f"2. Python版本: {sys.version}")
print(f"3. PyTorch版本: {torch.__version__}")

print("4. 检查预训练模型...")
if os.path.exists("pretrained_encoder_v2.pth"):
    print("   ✅ 预训练模型存在")
else:
    print("   ❌ 预训练模型不存在")

print("5. 检查数据目录...")
if os.path.exists("AU-ASD-TD-GASF-V2"):
    asd_files = len([f for f in os.listdir("AU-ASD-TD-GASF-V2/ASD") if f.endswith('.pt')])
    td_files = len([f for f in os.listdir("AU-ASD-TD-GASF-V2/TD") if f.endswith('.pt')])
    print(f"   ✅ 数据目录存在: ASD={asd_files}, TD={td_files}")
else:
    print("   ❌ 数据目录不存在")

print("6. 测试导入模块...")
try:
    from contrastive_model_v2 import MultiTaskEncoder, ClassificationHead
    print("   ✅ 导入模型成功")
except Exception as e:
    print(f"   ❌ 导入失败: {e}")

print("7. 测试加载预训练模型...")
try:
    from contrastive_model_v2 import MultiTaskEncoder
    encoder = MultiTaskEncoder(num_tasks=4, feature_dim=256, fusion_type='concat')
    state_dict = torch.load("pretrained_encoder_v2.pth", map_location='cpu', weights_only=True)
    encoder.load_state_dict(state_dict)
    print("   ✅ 加载预训练模型成功")
except Exception as e:
    print(f"   ❌ 加载失败: {e}")

print("8. 测试加载一个数据样本...")
try:
    sample_file = "AU-ASD-TD-GASF-V2/ASD/" + os.listdir("AU-ASD-TD-GASF-V2/ASD")[0]
    data = torch.load(sample_file, weights_only=True)
    print(f"   ✅ 加载数据成功: {list(data.keys())}")
    print(f"      数据形状: {[(k, v.shape) for k, v in data.items()]}")
except Exception as e:
    print(f"   ❌ 加载数据失败: {e}")

print("9. 测试前向传播...")
try:
    encoder.eval()
    # 添加batch维度
    data_batch = {k: v.unsqueeze(0) for k, v in data.items()}
    print(f"      添加batch后形状: {[(k, v.shape) for k, v in data_batch.items()]}")
    with torch.no_grad():
        feat = encoder(data_batch)
    print(f"   ✅ 前向传播成功，输出形状: {feat.shape}")
except Exception as e:
    print(f"   ❌ 前向传播失败: {e}")

print("\n✅ 调试完成！")
