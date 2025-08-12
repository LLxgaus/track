import cv2

def convert_video_max_height(input_video_path, output_video_path, max_height=500):
    # 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("无法打开输入视频文件！")
        return
    
    # 获取原始视频参数
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算等比例缩放后的尺寸
    if original_height <= max_height:
        print("视频高度已小于等于目标高度，无需缩放。")
        target_width, target_height = original_width, original_height
    else:
        scale_ratio = max_height / original_height
        target_height = max_height
        target_width = int(original_width * scale_ratio)
    
    print(f"原始分辨率: {original_width}x{original_height}")
    print(f"缩放后分辨率: {target_width}x{target_height}")
    
    # 创建VideoWriter对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式（MP4）
    out = cv2.VideoWriter(
        output_video_path, 
        fourcc, 
        fps, 
        (target_width, target_height)
    )
    
    # 逐帧处理视频
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 等比例缩放帧
        resized_frame = cv2.resize(
            frame, 
            (target_width, target_height), 
            interpolation=cv2.INTER_LINEAR
        )
        
        # 写入新视频
        out.write(resized_frame)
    
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"视频已等比例缩放（高度≤{max_height}px），保存至: {output_video_path}")

# 示例调用
input_video = "D:\\Track-Anything-master\\test_sample\\video3.mp4"    # 输入视频路径
output_video = "D:\\Track-Anything-master\\test_sample\\output_1080p.mp4"  # 输出视频路径
convert_video_max_height(input_video, output_video, max_height=1080)
