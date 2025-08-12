import gradio as gr
import os
import time
from datetime import datetime
import cv2
import numpy as np
import torch
from app_yolo import TrackingAnything, parse_augment

# 初始化参数
args = parse_augment()
args.port = 7860  # 修改为7860端口
args.device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 初始化模型
model = TrackingAnything(args.sam_checkpoint, args.xmem_checkpoint, args.e2fgvi_checkpoint, args)

# 定义结果保存路径
RESULT_BASE_PATH = r"D:\Track-Anything-master\result\track"

def create_result_dir():
    """创建带时间戳的结果目录"""
    os.makedirs(RESULT_BASE_PATH, exist_ok=True)  # 确保基础目录存在
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = os.path.join(RESULT_BASE_PATH, f"track_{timestamp}")
    os.makedirs(result_dir, exist_ok=True)
    return result_dir

def process_video(video_path, category_id):
    """处理视频并跟踪指定类别的目标"""
    # 创建结果目录
    result_dir = create_result_dir()
    print(f"结果将保存到: {result_dir}")  # 调试信息
    
    # 读取视频
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    
    # 提取纯数字的类别ID
    category_id = int(category_id.split("-")[0])
    
    # 重置模型状态
    model.reset_tracker()
    model.set_current_category(category_id)
    
    # 处理第一帧
    first_frame = frames[0]
    model.init_first_frame(first_frame)
    
    # 跟踪视频
    results = []
    for i, frame in enumerate(frames):
        result = model.track_frame(frame, category_id)  # 传入当前类别ID
        results.append(result)
        
        # 保存每一帧结果
        frame_path = os.path.join(result_dir, f"frame_{i:04d}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    
    # 生成结果视频
    output_video_path = os.path.join(result_dir, "tracking_result.mp4")
    generate_output_video(results, output_video_path)
    
    # 保存日志
    log_path = os.path.join(result_dir, "tracking_log.txt")
    with open(log_path, "w") as f:
        f.write(f"Tracking completed at {datetime.now()}\n")
        f.write(f"Video: {os.path.basename(video_path)}\n")
        f.write(f"Category: {category_id}\n")
        f.write(f"Total frames: {len(frames)}\n")
    
    return output_video_path, f"跟踪完成！结果已保存到: {result_dir}"

def generate_output_video(frames, output_path):
    """生成输出视频"""
    if not frames:
        return None
    
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (width, height))
    
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

# Gradio界面
with gr.Blocks(title="目标跟踪系统") as demo:
    gr.Markdown("# 🎯 目标跟踪系统")
    gr.Markdown("上传视频并选择目标类别，系统会自动跟踪不同类别的目标")
    
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(label="上传视频")
            category_dropdown = gr.Dropdown(
                choices=["1-默认类别", "2-人物", "3-车辆", "4-动物", "5-其他"],
                value="1-默认类别",
                label="选择目标类别"
            )
            submit_btn = gr.Button("开始跟踪", variant="primary")
        
        with gr.Column():
            output_video = gr.Video(label="跟踪结果")
            status_output = gr.Textbox(label="状态信息")
    
    # 按钮点击事件
    submit_btn.click(
        fn=process_video,
        inputs=[video_input, category_dropdown],
        outputs=[output_video, status_output],
        api_name="track_object"
    )

# 启动界面
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=args.port)