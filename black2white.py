import os
from PIL import Image
import numpy as np


def invert_image_colors(input_path, output_path):
    """
    将图片进行黑白颠倒（反色）处理
    """
    try:
        # 打开图片
        img = Image.open(input_path)

        # 转换为RGB模式（确保可以处理带透明度的PNG）
        if img.mode == 'RGBA':
            # 创建白色背景
            background = Image.new('RGB', img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])  # 3是alpha通道
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')

        # 使用numpy进行高效的反色处理
        img_array = np.array(img)
        inverted_array = 255 - img_array

        # 创建新的图像
        inverted_img = Image.fromarray(inverted_array.astype('uint8'))

        # 保存反色后的图片
        inverted_img.save(output_path)
        print(f"✓ 已处理: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
        return True
    except Exception as e:
        print(f"✗ 处理 {os.path.basename(input_path)} 时出错: {str(e)}")
        return False


def process_folder(folder_path):
    """
    处理指定文件夹中的所有PNG图片
    """
    if not os.path.exists(folder_path):
        print(f"错误: 文件夹路径不存在 - {folder_path}")
        return False

    if not os.path.isdir(folder_path):
        print(f"错误: 指定的路径不是文件夹 - {folder_path}")
        return False

    # 获取所有PNG文件
    png_files = [f for f in os.listdir(folder_path)
                 if f.lower().endswith('.png') and '_new.png' not in f.lower()]

    if not png_files:
        print("警告: 文件夹中没有找到PNG图片文件")
        return False

    print(f"找到 {len(png_files)} 个PNG图片文件，开始处理...")
    print("-" * 50)

    success_count = 0
    for filename in png_files:
        input_path = os.path.join(folder_path, filename)

        # 生成输出文件名（在原文件名后添加_new）
        name_parts = os.path.splitext(filename)
        output_filename = f"{name_parts[0]}_new{name_parts[1]}"
        output_path = os.path.join(folder_path, output_filename)

        if invert_image_colors(input_path, output_path):
            success_count += 1

    print("-" * 50)
    print(f"处理完成！成功处理 {success_count}/{len(png_files)} 个文件")
    return True


def main():
    """
    主程序：获取用户输入并执行处理
    """
    print("=== PNG图片黑白颠倒工具 ===")
    print("请输入要处理的文件夹路径（可以直接拖拽文件夹到此处）")
    print("示例: C:\\Users\\YourName\\Pictures 或 /home/user/pictures")
    print("-" * 60)

    # 获取用户输入的文件夹路径
    folder_path = input("文件夹路径: ").strip()

    # 处理用户可能拖拽文件夹的情况（去除引号）
    folder_path = folder_path.strip('"').strip("'")

    print("-" * 60)

    # 验证路径是否存在
    if not os.path.exists(folder_path):
        print(f"错误: 路径不存在 - {folder_path}")
        print("请检查路径是否正确，然后重新运行程序")
        return

    # 执行处理
    process_folder(folder_path)

    print("-" * 60)
    input("按回车键退出程序...")


if __name__ == "__main__":
    main()