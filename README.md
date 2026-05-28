# PyTorch 图像超分辨率大作业模板（SRResNet）

## 1. 环境安装
```bash
pip install -r requirements.txt
```

## 2. 数据准备
建议目录：
```text
data/
  DIV2K_train_HR/
  DIV2K_valid_HR/
```


## 2.1 自动下载 DIV2K（推荐）
```bash
python scripts/download_data.py --data_root data --split all
```

可选参数：
- `--split train|valid|all`：下载指定划分
- `--redownload`：强制重新下载
- `--clean_temp`：完成后删除临时压缩包和解压目录

下载完成后会自动校验图片数量（train>=800, valid>=100）。


## 2.2 一键初始化项目骨架（bash）
```bash
bash scripts/init_project.sh /d/sr_project
```

不传路径时默认初始化到 `/d/sr_project`：
```bash
bash scripts/init_project.sh
```

该脚本会自动创建目录并拷贝完整模板代码文件（train.py/test.py/模型/数据集脚本等）。

## 3. 训练
```bash
python train.py \
  --train_hr_dir data/DIV2K_train_HR \
  --val_hr_dir data/DIV2K_valid_HR \
  --scale 4 \
  --epochs 200 \
  --batch_size 16
```

## 4. 测试
```bash
python test.py \
  --test_hr_dir data/DIV2K_valid_HR \
  --model_path checkpoints/srresnet_x4/best_psnr.pth \
  --scale 4
```

## 5. 输出结果
- 模型权重：`checkpoints/srresnet_x4/`
- 重建图像：`results/images/`
- 训练日志：`checkpoints/srresnet_x4/history.json`

## 6. 达标建议
- 使用 DIV2K 训练，Set5/Set14 测试；
- 先确保 `L1 Loss` 收敛，再考虑加感知损失；
- 报告展示 PSNR、SSIM 以及至少 3 组可视化对比图。


## Initialize project (PowerShell)

If you are using Windows PowerShell, use the PowerShell initializer script:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/init_project.ps1
```

Custom target path:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/init_project.ps1 -ProjectRoot "D:\my_sr_project"
```
