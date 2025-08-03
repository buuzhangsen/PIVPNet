import argparse

parser = argparse.ArgumentParser()

# 训练控制参数
parser.add_argument('--resume_ckpt', type=str, default='',
                    help='[新添加]从指定检查点加载兼容权重（自动忽略不匹配参数）')
parser.add_argument('--epochs', type=int, default=300, 
                    help='最大训练轮数')
parser.add_argument('--batch_size', type=int, default=28,
                    help="每个GPU的批量大小")
parser.add_argument('--lr', type=float, default=3e-5, 
                    help='初始学习率')

# 数据参数
parser.add_argument('--de_type', nargs='+', default=['denoise_15','denoise_25','denoise_50', 'derain', 'dehaze', 'deblur', 'enhance'],
                    help='which type of degradations is training and testing for.')
parser.add_argument('--patch_size', type=int, default=128,
                    help='输入图像块大小')
parser.add_argument('--num_workers', type=int, default=12,
                    help='数据加载线程数')

# 路径参数
parser.add_argument('--data_file_dir', type=str, default='data_dir/',
                    help='基础数据目录')
parser.add_argument('--denoise_dir', type=str, 
                    default='data/Train/denoise/high/',
                    help='去噪训练数据路径')
parser.add_argument('--gopro_dir', type=str, 
                    default='data/Train/Deblur/',
                    help='去模糊训练数据路径')
parser.add_argument('--enhance_dir', type=str, 
                    default='data/Train/low_light_image_enhance/',
                    help='低光增强数据路径')
parser.add_argument('--derain_dir', type=str, 
                    default='data/Train/derain/',
                    help='去雨训练数据路径')
parser.add_argument('--dehaze_dir', type=str, 
                    default='data/Train/dehaze',
                    help='去雾训练数据路径')
# parser.add_argument('--enhance_path', type=str, default="data/test/low_lightimageenhance/", help='save path of test hazy images')
# parser.add_argument('--derain_path', type=str, default="data/test/derain/", help='save path of test raining images')
parser.add_argument('--output_path', type=str, default="output/",
                    help='输出文件路径')
parser.add_argument('--ckpt_path', type=str, default="ckpt/Denoise/",
                    help='检查点保存路径')

# 系统参数
parser.add_argument('--cuda', type=int, default=0,
                    help='[保留参数]CUDA设备ID')
parser.add_argument("--wblogger", type=str, default="",
                    help="Wandb项目名称（设为空禁用wandb）")
parser.add_argument("--ckpt_dir", type=str, default="",
                    help="检查点保存目录")
parser.add_argument("--num_gpus", type=int, default=2,
                    help="使用的GPU数量")

# 高级参数（保持默认即可）
parser.add_argument('--strict_resume', action='store_true',
                    help='[专家选项]严格恢复训练状态（需模型完全兼容）')
parser.add_argument('--resume_optimizer', action='store_true',
                    help='[专家选项]尝试恢复优化器状态')

options = parser.parse_args()
