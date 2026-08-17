import os
from PIL import Image
import argparse


def split_image(image_path, output_dir, rows=4, cols=4):
    """
    将单张图像分割为rows×cols块并保存（默认4x4）
    :param image_path: 输入图像路径
    :param output_dir: 输出目录
    :param rows: 纵向分割行数（竖为4）
    :param cols: 横向分割列数（横为4）
    """
    try:
        img = Image.open(image_path)
        width, height = img.size
        
        # 检查尺寸是否符合分割要求
        if width % cols != 0 or height % rows != 0:
            raise ValueError(f"图像尺寸必须能被{cols}（宽）和{rows}（高）整除，当前尺寸：{width}x{height}")
        
        block_width = width // cols
        block_height = height // rows
        
        # 创建图像专属输出文件夹
        img_filename = os.path.splitext(os.path.basename(image_path))[0]
        img_output_dir = os.path.join(output_dir, img_filename)
        os.makedirs(img_output_dir, exist_ok=True)
        
        for row in range(rows):
            for col in range(cols):
                x1 = col * block_width
                y1 = row * block_height
                x2 = x1 + block_width
                y2 = y1 + block_height
                
                block = img.crop((x1, y1, x2, y2))
                block_filename = f"block_row{row+1:02d}_col{col+1:02d}.jpg"  # 命名格式：块_行_列.jpg
                block.save(os.path.join(img_output_dir, block_filename))
        
        print(f"成功分割 {image_path} 为{rows*cols}块，保存于 {img_output_dir}")
    
    except Exception as e:
        print(f"处理 {image_path} 时出错：{str(e)}")


def main():
    parser = argparse.ArgumentParser(description="图像4x4均匀分割工具")
    parser.add_argument("--input_dir", default="/hy-tmp/QuadPriorformer_0/QuadpriorFormer/Enhancement/images", help="输入图像文件夹路径")
    parser.add_argument("--output_dir", default="/hy-tmp/QuadPriorformer_0/QuadpriorFormer/Enhancement/divide_result", help="输出结果文件夹路径")
    parser.add_argument("-r", "--rows", type=int, default=4, help="纵向分割块数（默认4）")
    parser.add_argument("-c", "--cols", type=int, default=4, help="横向分割块数（默认4）")
    
    args = parser.parse_args()
    
    # 检查输入目录是否存在
    if not os.path.isdir(args.input_dir):
        raise NotADirectoryError(f"输入目录不存在：{args.input_dir}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 支持的图像扩展名
    img_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')
    
    for filename in os.listdir(args.input_dir):
        file_path = os.path.join(args.input_dir, filename)
        if os.path.isfile(file_path) and filename.lower().endswith(img_extensions):
            split_image(file_path, args.output_dir, args.rows, args.cols)


if __name__ == "__main__":
    main()