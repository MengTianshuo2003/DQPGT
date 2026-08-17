import os
from PIL import Image, ImageOps
import argparse
import re


def combine_images(input_dir, output_path, rows=4, cols=4, margin=5, bg_color=(255, 255, 255)):
    """
    将分割后的 4x4 图像按顺序合成带间隔的矩阵图
    :param input_dir: 输入文件夹（包含子文件夹，每个子文件夹内存放分割后的块图像）
    :param output_path: 合成图像保存路径
    :param rows: 纵向块数（默认4）
    :param cols: 横向块数（默认4）
    :param margin: 图像间隔（像素，默认5）
    :param bg_color: 背景色（RGB元组，默认白色）
    """
    try:
        # 收集所有子文件夹（每个子文件夹对应一张原始图像的分割结果）
        subfolders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
        
        for subfolder in subfolders:
            folder_path = os.path.join(input_dir, subfolder)
            img_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            # 从文件名中提取行和列信息（格式：block_rowXX_colXX.jpg）
            img_data = []
            for filename in img_files:
                match = re.match(r'block_row(\d+)_col(\d+)\.(jpg|jpeg|png)', filename, re.IGNORECASE)
                if match:
                    row = int(match.group(1))
                    col = int(match.group(2))
                    img_data.append((row, col, os.path.join(folder_path, filename)))
            
            # 按行和列排序（行优先，列递增）
            img_data.sort(key=lambda x: (x[0], x[1]))
            
            # 检查是否有足够的图像
            if len(img_data) != rows * cols:
                raise ValueError(f"子文件夹 {subfolder} 中图像数量不符，需要{rows*cols}张，实际{len(img_data)}张")
            
            # 获取第一张图像尺寸（假设所有图像尺寸一致）
            first_img = Image.open(img_data[0][2])
            block_width, block_height = first_img.size
            
            # 计算合成图总尺寸
            total_width = block_width * cols + margin * (cols - 1)
            total_height = block_height * rows + margin * (rows - 1)
            
            # 创建空白合成图
            combined_img = Image.new('RGB', (total_width, total_height), bg_color)
            
            for idx, (row, col, img_path) in enumerate(img_data):
                # 计算当前块在合成图中的位置（行、列从1开始，间隔在块之间，四周无间隔）
                x = (col - 1) * (block_width + margin)
                y = (row - 1) * (block_height + margin)
                
                with Image.open(img_path) as img:
                    combined_img.paste(img, (x, y))
            
            # 保存合成图（文件名包含原文件夹名）
            output_filename = f"{subfolder}_combined.jpg"
            final_output_path = os.path.join(output_path, output_filename)
            combined_img.save(final_output_path)
            print(f"成功合成 {subfolder} 为 {final_output_path}")
    
    except Exception as e:
        print(f"合成图像时出错：{str(e)}")


def main():
    parser = argparse.ArgumentParser(description="4x4图像块合成工具")
    parser.add_argument("--input_dir", default="/hy-tmp/QuadPriorformer_0/QuadpriorFormer/Enhancement/divide_result", help="输入文件夹（包含分割后的子文件夹）")
    parser.add_argument("--output_dir", default="/hy-tmp/QuadPriorformer_0/QuadpriorFormer/Enhancement/combine_result", help="合成图像保存目录")
    parser.add_argument("-r", "--rows", type=int, default=4, help="纵向块数（默认4）")
    parser.add_argument("-c", "--cols", type=int, default=4, help="横向块数（默认4）")
    parser.add_argument("-m", "--margin", type=int, default=5, help="图像间隔（像素，默认5）")
    parser.add_argument("-b", "--bg_color", nargs=3, type=int, default=[255, 255, 255], 
                      help="背景色RGB值（默认白色，格式：255 255 255）")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 转换背景色为元组
    bg_color = tuple(args.bg_color)
    
    combine_images(
        input_dir=args.input_dir,
        output_path=args.output_dir,
        rows=args.rows,
        cols=args.cols,
        margin=args.margin,
        bg_color=bg_color
    )


if __name__ == "__main__":
    main()