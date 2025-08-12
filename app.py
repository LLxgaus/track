import gradio as gr
import argparse
import gdown
import cv2
import numpy as np
import os
import sys
sys.path.append(sys.path[0]+"/tracker")
sys.path.append(sys.path[0]+"/tracker/model")
from track_anything import TrackingAnything
from track_anything import parse_augment
import requests
import json
import torchvision
import torch 
from tools.painter import mask_painter
import psutil
import time
try: 
    from mmcv.cnn import ConvModule
except:
    os.system("mim install mmcv")
from fractions import Fraction
import datetime

def get_max_connected_bbox(mask):
    """从二值掩码中提取最大连通区域的边界框"""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    return (x, y, x+w, y+h)  # 返回(x1,y1,x2,y2)

def draw_bboxes_on_image(image, bboxes, colors=None):
    """在图像上绘制多个边界框（支持多类别）"""
    if colors is None:
        colors = [(0, 255, 0)] * len(bboxes)  # 默认绿色
    
    for idx, bbox in enumerate(bboxes):
        if bbox is None:
            continue
            
        color = colors[idx % len(colors)]
        x1, y1, x2, y2 = bbox
        
        # 绘制主体边界框
        thickness = 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # 绘制四个加粗的角落
        corner_length = min(int((x2 - x1) * 0.2), int((y2 - y1) * 0.2), 15)  # 角落长度不超过15像素
        thickness = 4
        
        # 左上角
        cv2.line(image, (x1, y1), (x1 + corner_length, y1), color, thickness)
        cv2.line(image, (x1, y1), (x1, y1 + corner_length), color, thickness)
        
        # 右上角
        cv2.line(image, (x2 - corner_length, y1), (x2, y1), color, thickness)
        cv2.line(image, (x2, y1), (x2, y1 + corner_length), color, thickness)
        
        # 左下角
        cv2.line(image, (x1, y2 - corner_length), (x1, y2), color, thickness)
        cv2.line(image, (x1, y2), (x1 + corner_length, y2), color, thickness)
        
        # 右下角
        cv2.line(image, (x2 - corner_length, y2), (x2, y2), color, thickness)
        cv2.line(image, (x2, y2 - corner_length), (x2, y2), color, thickness)
    
    return image

# download checkpoints
def download_checkpoint(url, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("download checkpoints ......")
        response = requests.get(url, stream=True)
        with open(filepath, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print("download successfully!")

    return filepath

def download_checkpoint_from_google_drive(file_id, folder, filename):
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, filename)

    if not os.path.exists(filepath):
        print("Downloading checkpoints from Google Drive... tips: If you cannot see the progress bar, please try to download it manuall \
              and put it in the checkpointes directory. E2FGVI-HQ-CVPR22.pth: https://github.com/MCG-NKU/E2FGVI(E2FGVI-HQ model)")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, filepath, quiet=False)
        print("Downloaded successfully!")

    return filepath

# convert points input to prompt state
def get_prompt(click_state, click_input):
    inputs = json.loads(click_input)
    points = click_state[0]
    labels = click_state[1]
    for input in inputs:
        points.append(input[:2])
        labels.append(input[2])
    click_state[0] = points
    click_state[1] = labels
    prompt = {
        "prompt_type":["click"],
        "input_point":click_state[0],
        "input_label":click_state[1],
        "multimask_output":"True",
    }
    return prompt


# extract frames from upload video
def get_frames_from_video(video_input, video_state):
    """
    Args:
        video_path:str
        timestamp:float64
    Return 
        [[0:nearest_frame], [nearest_frame:], nearest_frame]
    """
    video_path = video_input
    frames = []
    user_name = time.time()
    operation_log = [("",""),("Upload video already. Try click the image for adding targets to track and inpaint.","Normal")]
    try:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 获取原始视频尺寸
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 计算缩放比例（限制高度不超过720以获得更好的性能）
        max_height = 720  # 修改为720以获得更好的性能
        scale = min(1.0, max_height / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # 等比例缩放
                if scale < 1.0:
                    frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                current_memory_usage = psutil.virtual_memory().percent
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if current_memory_usage > 90:
                    operation_log = [("Memory usage is too high (>90%). Stop the video extraction. Please reduce the video resolution or frame rate.", "Error")]
                    print("Memory usage is too high (>90%). Please reduce the video resolution or frame rate.")
                    break
            else:
                break
    except (OSError, TypeError, ValueError, KeyError, SyntaxError) as e:
        print("read_frame_source:{} error. {}\n".format(video_path, str(e)))
    
    image_size = (frames[0].shape[0], frames[0].shape[1]) 
    # initialize video_state
    video_state = {
        "user_name": user_name,
        "video_name": os.path.split(video_path)[-1],
        "origin_images": frames,
        "painted_images": frames.copy(),
        "masks": [np.zeros((frames[0].shape[0],frames[0].shape[1]), np.uint8)]*len(frames),
        "logits": [None]*len(frames),
        "select_frame_number": 0,
        "fps": fps,
        "original_size": (original_height, original_width),  # 保存原始尺寸 (h,w)
        "scaled_size": (new_height, new_width),  # 保存缩放后尺寸 (h,w)
        "scale_ratio": scale  # 保存缩放比例
        }
    
    video_info = "Video Name: {}, FPS: {}, Total Frames: {}, Original Size: {}x{}, Scaled Size: {}x{}".format(
        video_state["video_name"], 
        video_state["fps"], 
        len(frames),
        video_state["original_size"][1],  # width
        video_state["original_size"][0],  # height
        video_state["scaled_size"][1],    # scaled width
        video_state["scaled_size"][0]    # scaled height
    )
    
    model.samcontroler.sam_controler.reset_image() 
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][0])
    return video_state, video_info, video_state["origin_images"][0], gr.update(visible=True, maximum=len(frames), value=1), gr.update(visible=True, maximum=len(frames), value=len(frames)), \
                        gr.update(visible=True),\
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True), gr.update(visible=True), \
                        gr.update(visible=True, value=operation_log)

def run_example(example):
    return video_input
# get the select frame from gradio slider
def select_template(image_selection_slider, video_state, interactive_state, mask_dropdown):

    # images = video_state[1]
    image_selection_slider -= 1
    video_state["select_frame_number"] = image_selection_slider

    # once select a new template frame, set the image in sam

    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][image_selection_slider])

    # update the masks when select a new template frame
    # if video_state["masks"][image_selection_slider] is not None:
        # video_state["painted_images"][image_selection_slider] = mask_painter(video_state["origin_images"][image_selection_slider], video_state["masks"][image_selection_slider])
    if mask_dropdown:
        print("ok")
    operation_log = [("",""), ("Select frame {}. Try click image and add mask for tracking.".format(image_selection_slider),"Normal")]


    return video_state["painted_images"][image_selection_slider], video_state, interactive_state, operation_log

# set the tracking end frame
def get_end_number(track_pause_number_slider, video_state, interactive_state):
    interactive_state["track_end_number"] = track_pause_number_slider
    operation_log = [("",""),("Set the tracking finish at frame {}".format(track_pause_number_slider),"Normal")]

    return video_state["painted_images"][track_pause_number_slider],interactive_state, operation_log

def get_resize_ratio(resize_ratio_slider, interactive_state):
    interactive_state["resize_ratio"] = resize_ratio_slider

    return interactive_state

# use sam to get the mask
def sam_refine(video_state, point_prompt, click_state, interactive_state, evt:gr.SelectData):
    """
    Args:
        template_frame: PIL.Image
        point_prompt: flag for positive or negative button click
        click_state: [[points], [labels]]
    """
    if point_prompt == "Positive":
        coordinate = "[[{},{},1]]".format(evt.index[0], evt.index[1])
        interactive_state["positive_click_times"] += 1
    else:
        coordinate = "[[{},{},0]]".format(evt.index[0], evt.index[1])
        interactive_state["negative_click_times"] += 1
    
    # prompt for sam model
    model.samcontroler.sam_controler.reset_image()
    model.samcontroler.sam_controler.set_image(video_state["origin_images"][video_state["select_frame_number"]])
    prompt = get_prompt(click_state=click_state, click_input=coordinate)

    mask, logit, painted_image = model.first_frame_click( 
                                                      image=video_state["origin_images"][video_state["select_frame_number"]], 
                                                      points=np.array(prompt["input_point"]),
                                                      labels=np.array(prompt["input_label"]),
                                                      multimask=prompt["multimask_output"],
                                                      )
    video_state["masks"][video_state["select_frame_number"]] = mask
    video_state["logits"][video_state["select_frame_number"]] = logit
    video_state["painted_images"][video_state["select_frame_number"]] = painted_image

    operation_log = [("",""), ("Use SAM for segment. You can try add positive and negative points by clicking. Or press Clear clicks button to refresh the image. Press Add mask button when you are satisfied with the segment","Normal")]
    return painted_image, video_state, interactive_state, operation_log

def add_multi_mask(video_state, interactive_state, mask_dropdown):
    try:
        mask = video_state["masks"][video_state["select_frame_number"]]
        interactive_state["multi_mask"]["masks"].append(mask)
        interactive_state["multi_mask"]["mask_names"].append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        mask_dropdown.append("mask_{:03d}".format(len(interactive_state["multi_mask"]["masks"])))
        select_frame, run_status = show_mask(video_state, interactive_state, mask_dropdown)

        operation_log = [("",""),("Added a mask, use the mask select for target tracking or inpainting.","Normal")]
    except:
        operation_log = [("Please click the left image to generate mask.", "Error"), ("","")]
    return interactive_state, gr.update(choices=interactive_state["multi_mask"]["mask_names"], value=mask_dropdown), select_frame, [[],[]], operation_log

def clear_click(video_state, click_state):
    click_state = [[],[]]
    template_frame = video_state["origin_images"][video_state["select_frame_number"]]
    operation_log = [("",""), ("Clear points history and refresh the image.","Normal")]
    return template_frame, click_state, operation_log

def remove_multi_mask(interactive_state, mask_dropdown):
    interactive_state["multi_mask"]["mask_names"]= []
    interactive_state["multi_mask"]["masks"] = []

    operation_log = [("",""), ("Remove all mask, please add new masks","Normal")]
    return interactive_state, gr.update(choices=[],value=[]), operation_log

def show_mask(video_state, interactive_state, mask_dropdown):
    mask_dropdown.sort()
    select_frame = video_state["origin_images"][video_state["select_frame_number"]]
    for i in range(len(mask_dropdown)):
        mask_number = int(mask_dropdown[i].split("_")[1]) - 1
        mask = interactive_state["multi_mask"]["masks"][mask_number]
        select_frame = mask_painter(select_frame, mask.astype('uint8'), mask_color=mask_number+2)
    
    operation_log = [("",""), ("Select {} for tracking or inpainting".format(mask_dropdown),"Normal")]
    return select_frame, operation_log

# 添加类别分类功能函数
def add_class(interactive_state, class_name):
    """Add a new class to the tracking system"""
    if class_name.strip() == "":
        return interactive_state, "Class name cannot be empty!"
    
    # Initialize class system if not exists
    if "classes" not in interactive_state:
        interactive_state["classes"] = {}
        next_class_id = 0
    else:
        next_class_id = max(interactive_state["classes"].keys(), default=-1) + 1
    
    # Add new class
    interactive_state["classes"][next_class_id] = {
        "name": class_name,
        "mask_ids": []  # Will store mask IDs assigned to this class
    }
    
    return interactive_state, f"Added class: {class_name} (ID: {next_class_id})"

def assign_class(interactive_state, mask_name, class_id):
    """Assign a mask to a class"""
    # Validate input
    if mask_name == "" or class_id == "":
        return interactive_state, "Please select both mask and class"
    
    try:
        class_id = int(class_id)
    except:
        return interactive_state, "Invalid class ID"
    
    # Find mask ID from mask_name
    mask_number = int(mask_name.split("_")[1])
    mask_id = mask_number - 1
    
    # Ensure classes exist in state
    if "classes" not in interactive_state:
        return interactive_state, "No classes exist. Please add classes first."
    
    # Ensure class exists
    if class_id not in interactive_state["classes"]:
        return interactive_state, f"Class ID {class_id} does not exist"
    
    # Remove mask from any previous class
    for cid, class_info in interactive_state["classes"].items():
        if mask_id in class_info["mask_ids"]:
            class_info["mask_ids"].remove(mask_id)
    
    # Add mask to new class
    interactive_state["classes"][class_id]["mask_ids"].append(mask_id)
    
    return interactive_state, f"Assigned {mask_name} to class {interactive_state['classes'][class_id]['name']}"

# 更新类显示函数
def update_class_dropdown(interactive_state):
    """Update class selection dropdown"""
    class_options = []
    if "classes" in interactive_state:
        for class_id, class_info in interactive_state["classes"].items():
            class_options.append((f"{class_info['name']} (ID: {class_id})", class_id))
    return gr.update(choices=[option[0] for option in class_options])

def update_class_list(interactive_state):
    """Generate HTML for class assignments display"""
    if "classes" not in interactive_state or not interactive_state["classes"]:
        return "<p>No classes added yet</p>"
    
    html = "<div style='max-height: 200px; overflow-y: auto;'>"
    for class_id, class_info in interactive_state["classes"].items():
        mask_list = ", ".join([f"mask_{mid+1:03d}" for mid in class_info["mask_ids"]]) or "None"
        html += f"<p><b>{class_info['name']}</b> (ID: {class_id}): {mask_list}</p>"
    html += "</div>"
    return html

# tracking vos
def vos_tracking_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Track the selected masks, and then you can select the masks for inpainting.","Normal")]
    model.xmem.clear_memory()
    
    # [1] 获取要追踪的帧范围
    if interactive_state["track_end_number"]:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:interactive_state["track_end_number"]]
    else:
        following_frames = video_state["origin_images"][video_state["select_frame_number"]:]
    
    # [2] 处理多掩码选择
    if interactive_state["multi_mask"]["masks"]:
        if len(mask_dropdown) == 0:
            mask_dropdown = ["mask_001"]
        mask_dropdown.sort()
        template_mask = interactive_state["multi_mask"]["masks"][int(mask_dropdown[0].split("_")[1]) - 1] * (int(mask_dropdown[0].split("_")[1]))
        for i in range(1,len(mask_dropdown)):
            mask_number = int(mask_dropdown[i].split("_")[1]) - 1 
            template_mask = np.clip(template_mask+interactive_state["multi_mask"]["masks"][mask_number]*(mask_number+1), 0, mask_number+1)
        video_state["masks"][video_state["select_frame_number"]] = template_mask
    else:      
        template_mask = video_state["masks"][video_state["select_frame_number"]]
    
    # [3] 检查掩码有效性
    if len(np.unique(template_mask))==1:
        template_mask[0][0]=1
        operation_log = [("Error! Please add at least one mask to track by clicking the left image.","Error"), ("","")]
        return None, video_state, interactive_state, operation_log
    
    # [4] 运行追踪模型
    masks, logits, painted_images = model.generator(images=following_frames, template_mask=template_mask)
    model.xmem.clear_memory()

    # [5] 更新视频状态
    if interactive_state["track_end_number"]: 
        video_state["masks"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = masks
        video_state["logits"][video_state["select_frame_number"]:interactive_state["track_end_number"]] = logits
    else:
        video_state["masks"][video_state["select_frame_number"]:] = masks
        video_state["logits"][video_state["select_frame_number"]:] = logits
    
    # [6] 创建COCO格式的输出目录
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("./result/track", timestamp)
    coco_dir = os.path.join(output_dir, "coco_data")
    os.makedirs(os.path.join(coco_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(coco_dir, "annotations"), exist_ok=True)
    
    # [7] 准备COCO数据结构
    coco_data = {
        "info": {
            "description": "COCO dataset generated by Track-Anything",
            "url": "",
            "version": "1.0",
            "year": datetime.datetime.now().year,
            "contributor": "",
            "date_created": datetime.datetime.now().strftime("%Y/%m/%d")
        },
        "licenses": [{
            "url": "",
            "id": 1,
            "name": "Unknown"
        }],
        "images": [],
        "annotations": [],
        "categories": []
    }
    
    # [8] 创建类别映射
    category_map = {}
    if "classes" in interactive_state:
        for class_id, class_info in sorted(interactive_state["classes"].items()):
            category_id = len(coco_data["categories"]) + 1  # COCO类别ID从1开始
            category_map[class_id] = category_id
            coco_data["categories"].append({
                "id": category_id,
                "name": class_info["name"],
                "supercategory": "object"
            })
    
    # 为未分类的对象添加默认类别
    if not coco_data["categories"]:
        category_map[0] = 1
        coco_data["categories"].append({
            "id": 1,
            "name": "unknown",
            "supercategory": "object"
        })
    
    # [9] 实例跟踪和颜色映射
    instance_counter = 1  # 实例ID从1开始
    instance_colors = {}  # 存储每个实例的颜色
    scale_ratio = 1.0 / video_state["scale_ratio"] if video_state["scale_ratio"] < 1.0 else 1.0
    original_h, original_w = video_state["original_size"]
    
    # [10] 处理每一帧
    bboxes_list = []
    for frame_idx, mask in enumerate(video_state["masks"]):
        if mask is None:
            bboxes_list.append([])
            continue
        
        # 获取当前帧
        if frame_idx < video_state["select_frame_number"]:
            frame = video_state["origin_images"][frame_idx]
        else:
            frame = following_frames[frame_idx - video_state["select_frame_number"]]
        
        # 缩放回原始分辨率
        if video_state["scale_ratio"] < 1.0:
            original_frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                (original_w, original_h),
                interpolation=cv2.INTER_CUBIC
            )
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        else:
            original_frame = frame.copy()
        
        # 保存图像到COCO目录
        img_filename = f"frame_{frame_idx:04d}.jpg"
        img_path = os.path.join(coco_dir, "images", img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR))
        
        # 添加图像信息到COCO数据
        image_id = frame_idx + 1  # COCO图像ID从1开始
        coco_data["images"].append({
            "id": image_id,
            "width": original_w,
            "height": original_h,
            "file_name": img_filename,
            "license": 1,
            "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        # 处理每个mask的边界框
        unique_labels = np.unique(mask)[1:]  # 忽略背景0
        frame_bboxes = []
        
        for label in unique_labels:
            binary_mask = (mask == label).astype(np.uint8)
            bbox = get_max_connected_bbox(binary_mask)
            if not bbox:
                continue
                
            # 关键修改：在缩放后的坐标上计算边界框，然后转换到原始分辨率
            x1, y1, x2, y2 = bbox
            # 首先计算缩放后的实际像素坐标
            scaled_x1 = x1 * scale_ratio
            scaled_y1 = y1 * scale_ratio
            scaled_x2 = x2 * scale_ratio
            scaled_y2 = y2 * scale_ratio
            
            # 转换为COCO格式的bbox [x,y,width,height]
            coco_bbox = [
                scaled_x1,  # x
                scaled_y1,  # y
                scaled_x2 - scaled_x1,  # width
                scaled_y2 - scaled_y1   # height
            ]
            
            # 获取类别信息
            mask_id = label - 1  # mask编号从1开始，调整为0开始
            class_id = 0  # 默认类别
            class_name = "unknown"
            
            # 查找mask对应的类别
            if "classes" in interactive_state:
                for cid, class_info in interactive_state["classes"].items():
                    if mask_id in class_info["mask_ids"]:
                        class_id = cid
                        class_name = class_info["name"]
                        break
            
            # 为每个新实例分配唯一ID和颜色
            instance_key = f"{class_id}_{mask_id}"
            if instance_key not in instance_colors:
                # 使用HSV颜色空间均匀分布颜色
                hue = (instance_counter * 30) % 180  # 每30度一个颜色，避免太接近
                color = cv2.cvtColor(np.uint8([[[hue, 255, 220]]]), cv2.COLOR_HSV2BGR)[0][0]
                color = (int(color[0]), int(color[1]), int(color[2]))
                instance_colors[instance_key] = {
                    "id": instance_counter,
                    "color": color
                }
                instance_counter += 1
            
            instance_info = instance_colors[instance_key]
            
            # 添加注释到COCO数据
            coco_data["annotations"].append({
                "id": len(coco_data["annotations"]) + 1,
                "image_id": image_id,
                "category_id": category_map.get(class_id, 1),  # 默认为第一个类别
                "segmentation": [],  # 可以添加分割信息，这里简化为空
                "area": coco_bbox[2] * coco_bbox[3],  # width * height
                "bbox": coco_bbox,
                "iscrowd": 0,
                "instance_id": instance_info["id"]
            })
            
            # 保存原始坐标和实例信息用于绘制
            bbox_original = (
                int(scaled_x1),
                int(scaled_y1),
                int(scaled_x2),
                int(scaled_y2)
            )
            frame_bboxes.append({
                "bbox": bbox_original,
                "class_name": class_name,
                "instance_id": instance_info["id"],
                "color": instance_info["color"]
            })
        
        bboxes_list.append(frame_bboxes)
    
    # [11] 保存COCO格式的JSON文件
    coco_json_path = os.path.join(coco_dir, "annotations", "instances.json")
    with open(coco_json_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    # [12] 绘制带类别名称和实例ID的边界框（使用改进后的样式）
    video_state["painted_images"] = []
    for idx, frame in enumerate(video_state["origin_images"]):
        # 获取原始分辨率帧
        if video_state["scale_ratio"] < 1.0:
            original_frame = cv2.resize(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
                (original_w, original_h),
                interpolation=cv2.INTER_CUBIC
            )
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        else:
            original_frame = frame.copy()
        
        if idx < video_state["select_frame_number"]:
            painted_frame = original_frame.copy()
        else:
            frame_idx = idx - video_state["select_frame_number"]
            if frame_idx < len(bboxes_list):
                painted_frame = original_frame.copy()
                for obj in bboxes_list[frame_idx]:
                    bbox = obj["bbox"]
                    color = obj["color"]
                    class_name = obj["class_name"]
                    instance_id = obj["instance_id"]
                    
                    # 绘制边界框（使用改进后的样式）
                    x1, y1, x2, y2 = bbox
                    
                    # 绘制主体边界框
                    thickness = 2
                    cv2.rectangle(painted_frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # 绘制四个加粗的角落
                    corner_length = min(int((x2 - x1) * 0.2), int((y2 - y1) * 0.2), 15)
                    thickness = 8
                    
                    # 左上角
                    cv2.line(painted_frame, (x1, y1), (x1 + corner_length, y1), color, thickness)
                    cv2.line(painted_frame, (x1, y1), (x1, y1 + corner_length), color, thickness)
                    
                    # 右上角
                    cv2.line(painted_frame, (x2 - corner_length, y1), (x2, y1), color, thickness)
                    cv2.line(painted_frame, (x2, y1), (x2, y1 + corner_length), color, thickness)
                    
                    # 左下角
                    cv2.line(painted_frame, (x1, y2 - corner_length), (x1, y2), color, thickness)
                    cv2.line(painted_frame, (x1, y2), (x1 + corner_length, y2), color, thickness)
                    
                    # 右下角
                    cv2.line(painted_frame, (x2 - corner_length, y2), (x2, y2), color, thickness)
                    cv2.line(painted_frame, (x2, y2 - corner_length), (x2, y2), color, thickness)
                    
                    # 添加类别和实例ID标签（改进后的样式）
                    label = f"{instance_id}-{class_name}"
                    font_scale = 1.6  # 增大字体
                    thickness = 2    # 加粗字体
                    
                    # 计算文本大小
                    (label_width, label_height), baseline = cv2.getTextSize(
                        label, 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        font_scale, 
                        thickness
                    )
                    
                    # 调整标签位置（放在边界框上方）
                    text_x = x1
                    text_y = y1 - 5 if y1 - 5 > label_height else y1 + label_height + 5
                    
                    # 绘制文字（无背景，直接绘制）
                    cv2.putText(painted_frame, label,
                               (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale,
                               color,  # 使用与边界框相同的颜色
                               thickness, 
                               cv2.LINE_AA)
            else:
                painted_frame = original_frame.copy()
        
        video_state["painted_images"].append(painted_frame)
    
    # [13] 生成输出视频
    video_output = generate_video_from_frames(
        video_state["painted_images"], 
        output_path=os.path.join(output_dir, video_state["video_name"]), 
        fps=float(video_state["fps"])
    )
    
    operation_log.append(("", f"COCO格式数据已保存到: {coco_dir}"))
    return video_output, video_state, interactive_state, operation_log

# inpaint 
def inpaint_video(video_state, interactive_state, mask_dropdown):
    operation_log = [("",""), ("Removed the selected masks.","Normal")]

    frames = np.asarray(video_state["origin_images"])
    fps = video_state["fps"]
    inpaint_masks = np.asarray(video_state["masks"])
    if len(mask_dropdown) == 0:
        mask_dropdown = ["mask_001"]
    mask_dropdown.sort()
    # convert mask_dropdown to mask numbers
    inpaint_mask_numbers = [int(mask_dropdown[i].split("_")[1]) for i in range(len(mask_dropdown))]
    # interate through all masks and remove the masks that are not in mask_dropdown
    unique_masks = np.unique(inpaint_masks)
    num_masks = len(unique_masks) - 1
    for i in range(1, num_masks + 1):
        if i in inpaint_mask_numbers:
            continue
        inpaint_masks[inpaint_masks==i] = 0
    # inpaint for videos

    try:
        inpainted_frames = model.baseinpainter.inpaint(frames, inpaint_masks, ratio=interactive_state["resize_ratio"])   # numpy array, T, H, W, 3
    except:
        operation_log = [("Error! You are trying to inpaint without masks input. Please track the selected mask first, and then press inpaint. If VRAM exceeded, please use the resize ratio to scaling down the image size.","Error"), ("","")]
        inpainted_frames = video_state["origin_images"]
    video_output = generate_video_from_frames(inpainted_frames, output_path="./result/inpaint/{}".format(video_state["video_name"]), fps=fps) # import video_input to name the output video

    return video_output, operation_log


# generate video after vos inference
def generate_video_from_frames(frames, output_path, fps=30):
    """
    Generates a video from a list of frames.
    
    Args:
        frames (list of numpy arrays): The frames to include in the video.
        output_path (str): The path to save the generated video.
        fps (int, optional): The frame rate of the output video. Defaults to 30.
    """
    # 确保 fps 是 Python 原生的 float 或 Fraction
    if isinstance(fps, (np.float32, np.float64)):
        fps = float(fps)  # 转换为 Python 原生 float
    elif isinstance(fps, (int, float)):
        fps = Fraction(fps).limit_denominator()  # 转换为有理数，例如 29.97 → 30000/1001
    else:
        fps = Fraction(fps)  # 其他情况尝试直接转换
    
    # 确保 frames 是 numpy 数组
    frames = np.asarray(frames)
    
    # 确保帧数据是 uint8 类型
    if frames.dtype != np.uint8:
        frames = (frames * 255).clip(0, 255).astype(np.uint8)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 获取视频尺寸
    height, width = frames[0].shape[:2]
    
    # 使用 OpenCV 写入视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 或者使用 'avc1'
    video_writer = cv2.VideoWriter(output_path, fourcc, float(fps), (width, height))
    
    for frame in frames:
        # 确保颜色通道顺序是 BGR (OpenCV 的要求)
        if frame.shape[2] == 3:  # RGB 转 BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    
    video_writer.release()
    return output_path


# args, defined in track_anything.py
args = parse_augment()

# check and download checkpoints if needed
SAM_checkpoint_dict = {
    'vit_h': "sam_vit_h_4b8939.pth",
    'vit_l': "sam_vit_l_0b3195.pth", 
    "vit_b": "sam_vit_b_01ec64.pth"
}
SAM_checkpoint_url_dict = {
    'vit_h': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    'vit_l': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    'vit_b': "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
sam_checkpoint = SAM_checkpoint_dict[args.sam_model_type] 
sam_checkpoint_url = SAM_checkpoint_url_dict[args.sam_model_type] 
xmem_checkpoint = "XMem-s012.pth"
xmem_checkpoint_url = "https://github.com/hkchengrex/XMem/releases/download/v1.0/XMem-s012.pth"
e2fgvi_checkpoint = "E2FGVI-HQ-CVPR22.pth"
e2fgvi_checkpoint_id = "10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3"


folder ="./checkpoints"
SAM_checkpoint = download_checkpoint(sam_checkpoint_url, folder, sam_checkpoint)
xmem_checkpoint = download_checkpoint(xmem_checkpoint_url, folder, xmem_checkpoint)
e2fgvi_checkpoint = download_checkpoint_from_google_drive(e2fgvi_checkpoint_id, folder, e2fgvi_checkpoint)
args.port = 12212
args.device = "cuda:0"
# args.mask_save = True

# initialize sam, xmem, e2fgvi models
model = TrackingAnything(SAM_checkpoint, xmem_checkpoint, e2fgvi_checkpoint,args)


title = """<p><h1 align="center">Track-Anything</h1></p>
    """
description = """<p>Gradio demo for Track Anything, a flexible and interactive tool for video object tracking, segmentation, and inpainting. I To use it, simply upload your video, or click one of the examples to load them. Code: <a href="https://github.com/gaomingqi/Track-Anything">https://github.com/gaomingqi/Track-Anything</a> <a href="https://huggingface.co/spaces/watchtowerss/Track-Anything?duplicate=true"><img style="display: inline; margin-top: 0em; margin-bottom: 0em" src="https://bit.ly/3gLdBN6" alt="Duplicate Space" /></a></p>"""


with gr.Blocks() as iface:
    """
        state for 
    """
    click_state = gr.State([[],[]])
    interactive_state = gr.State({
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": None,
        "resize_ratio": 1,
        # 添加类别分类状态
        "classes": {}
    }
    )

    video_state = gr.State(
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        }
    )
    gr.Markdown(title)
    gr.Markdown(description)
    with gr.Row():

        # for user video input
        with gr.Column():
            with gr.Row(scale=0.4):
                video_input = gr.Video(autosize=True)
                with gr.Column():
                    video_info = gr.Textbox(label="Video Info")
                    resize_info = gr.Textbox(value="If you want to use the inpaint function, it is best to git clone the repo and use a machine with more VRAM locally. \
                                            Alternatively, you can use the resize ratio slider to scale down the original image to around 360P resolution for faster processing.", label="Tips for running this demo.")
                    resize_ratio_slider = gr.Slider(minimum=0.02, maximum=1, step=0.02, value=1, label="Resize ratio", visible=True)
          

            with gr.Row():
                # put the template frame under the radio button
                with gr.Column():
                    # extract frames
                    with gr.Column():
                        extract_frames_button = gr.Button(value="Get video info", interactive=True, variant="primary") 

                     # click points settins, negative or positive, mode continuous or single
                    with gr.Row():
                        with gr.Row():
                            point_prompt = gr.Radio(
                                choices=["Positive",  "Negative"],
                                value="Positive",
                                label="Point prompt",
                                interactive=True,
                                visible=False)
                            remove_mask_button = gr.Button(value="Remove mask", interactive=True, visible=False) 
                            clear_button_click = gr.Button(value="Clear clicks", interactive=True, visible=False).style(height=160)
                            Add_mask_button = gr.Button(value="Add mask", interactive=True, visible=False)
                    template_frame = gr.Image(type="pil",interactive=True, elem_id="template_frame", visible=False).style(height=360)
                    image_selection_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track start frame", visible=False)
                    track_pause_number_slider = gr.Slider(minimum=1, maximum=100, step=1, value=1, label="Track end frame", visible=False)
            
                with gr.Column():
                    run_status = gr.HighlightedText(value=[("Text","Error"),("to be","Label 2"),("highlighted","Label 3")], visible=False)
                    mask_dropdown = gr.Dropdown(multiselect=True, value=[], label="Mask selection", info=".", visible=False)
                    video_output = gr.Video(autosize=True, visible=False).style(height=360)
                    with gr.Row():
                        tracking_video_predict_button = gr.Button(value="Tracking", visible=False)
                        inpaint_video_predict_button = gr.Button(value="Inpainting", visible=False)
    
    # === 添加类别分类区域 ===
    with gr.Row(visible=False) as class_ui:
        with gr.Column():
            gr.Markdown("## Object Classification")
            with gr.Row():
                class_name_input = gr.Textbox(label="New Class Name", 
                                             placeholder="Enter class name (e.g., 'Person', 'Car')")
                add_class_button = gr.Button("Add Class")
                
            class_status = gr.Textbox(label="Classification Status", interactive=False)
            
            with gr.Row():
                # 修改为多选下拉菜单
                class_dropdown = gr.Dropdown([], label="Select Class", interactive=True, multiselect=True)
                mask_selection_dropdown = gr.Dropdown([], label="Select Mask", interactive=True, multiselect=True)
                assign_button = gr.Button("Assign Class to Mask")
            
            gr.Markdown("### Current Class Assignments")
            class_table = gr.HTML("<p>No classes added yet</p>", label="Assigned Classes")

    # first step: get the video information 
    extract_frames_button.click(
        fn=get_frames_from_video,
        inputs=[
            video_input, video_state
        ],
        outputs=[video_state, video_info, template_frame,
                 image_selection_slider, track_pause_number_slider,point_prompt, clear_button_click, Add_mask_button, template_frame,
                 tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button, inpaint_video_predict_button, run_status]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[class_ui]
    )

    # second step: select images from slider
    image_selection_slider.release(fn=select_template, 
                                   inputs=[image_selection_slider, video_state, interactive_state, mask_dropdown], 
                                   outputs=[template_frame, video_state, interactive_state, run_status], api_name="select_image")
    track_pause_number_slider.release(fn=get_end_number, 
                                   inputs=[track_pause_number_slider, video_state, interactive_state], 
                                   outputs=[template_frame, interactive_state, run_status], api_name="end_image")
    resize_ratio_slider.release(fn=get_resize_ratio, 
                                   inputs=[resize_ratio_slider, interactive_state], 
                                   outputs=[interactive_state], api_name="resize_ratio")
    
    # click select image to get mask using sam
    template_frame.select(
        fn=sam_refine,
        inputs=[video_state, point_prompt, click_state, interactive_state],
        outputs=[template_frame, video_state, interactive_state, run_status]
    )

    # add different mask
    Add_mask_button.click(
        fn=add_multi_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, template_frame, click_state, run_status]
    )

    remove_mask_button.click(
        fn=remove_multi_mask,
        inputs=[interactive_state, mask_dropdown],
        outputs=[interactive_state, mask_dropdown, run_status]
    )

    # tracking video from select image and mask
    tracking_video_predict_button.click(
        fn=vos_tracking_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, video_state, interactive_state, run_status]
    )

    # inpaint video from select image and mask
    inpaint_video_predict_button.click(
        fn=inpaint_video,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[video_output, run_status]
    )

    # click to get mask
    mask_dropdown.change(
        fn=show_mask,
        inputs=[video_state, interactive_state, mask_dropdown],
        outputs=[template_frame, run_status] 
    ).then(
        fn=update_class_list,
        inputs=[interactive_state],
        outputs=[class_table]
    )
    
    # === 类别分类事件处理 ===
    # 添加新类别
    add_class_button.click(
        fn=add_class,
        inputs=[interactive_state, class_name_input],
        outputs=[interactive_state, class_status]
    ).then(
        fn=update_class_dropdown,
        inputs=[interactive_state],
        outputs=[class_dropdown]
    ).then(
        fn=update_class_list,
        inputs=[interactive_state],
        outputs=[class_table]
    )
    # 分配类别到掩码 - 修改后的处理函数
    def assign_classes(interactive_state, mask_names, class_selections):
        """处理多选分配"""
        if not mask_names or not class_selections:
            return interactive_state, "Please select both masks and classes"
        
        # 确保classes存在
        if "classes" not in interactive_state:
            return interactive_state, "No classes exist. Please add classes first."
        
        # 从选择中提取类ID
        class_ids = []
        for selection in class_selections:
            try:
                # 从格式"name (ID: x)"中提取ID
                class_id = int(selection.split("(ID: ")[1].rstrip(")"))
                class_ids.append(class_id)
            except (IndexError, ValueError):
                return interactive_state, f"Invalid class selection format: {selection}"
    
        # 处理每个选中的mask
        for mask_name in mask_names:
            try:
                mask_id = int(mask_name.split("_")[1]) - 1
            except (IndexError, ValueError):
                return interactive_state, f"Invalid mask name format: {mask_name}"
            
            # 先移除该mask的所有现有分配
            for class_info in interactive_state["classes"].values():
                if mask_id in class_info["mask_ids"]:
                    class_info["mask_ids"].remove(mask_id)
        
            # 添加到新选中的classes
            for class_id in class_ids:
                if class_id in interactive_state["classes"]:
                    if mask_id not in interactive_state["classes"][class_id]["mask_ids"]:
                        interactive_state["classes"][class_id]["mask_ids"].append(mask_id)
    
        return interactive_state, f"Assigned {len(mask_names)} masks to {len(class_ids)} classes"
        
        
    # 分配类别到掩码
    assign_button.click(
        fn=assign_classes,
        inputs=[interactive_state, mask_selection_dropdown, class_dropdown],
        outputs=[interactive_state, class_status]
    ).then(
        fn=update_class_list,
        inputs=[interactive_state],
        outputs=[class_table]
    )
    
    # 更新update_class_list函数以正确显示分配
    def update_class_list(interactive_state):
        """Generate HTML for class assignments display"""
        if "classes" not in interactive_state or not interactive_state["classes"]:
            return "<p>No classes added yet</p>"
    
        # 生成HTML
        html = "<div style='max-height: 200px; overflow-y: auto;'>"
        for class_id, class_info in sorted(interactive_state["classes"].items()):
            if "mask_ids" not in class_info:
                class_info["mask_ids"] = []
            mask_list = ", ".join([f"mask_{mid+1:03d}" for mid in sorted(class_info["mask_ids"])]) or "None"
            html += f"<p><b>{class_info['name']}</b> (ID: {class_id}): {mask_list}</p>"
        html += "</div>"
        return html

    # 更新update_class_dropdown以支持多选

    def update_class_dropdown(interactive_state):
        """Update class selection dropdown with both name and ID"""
        if "classes" not in interactive_state:
            return gr.update(choices=[])
    
        choices = []
        for class_id, class_info in interactive_state["classes"].items():
            display_text = f"{class_info['name']} (ID: {class_id})"
            choices.append((display_text, display_text))  # 保持显示文本和值一致
    
        return gr.update(choices=choices)

    # 更新掩码选择下拉菜单
    mask_dropdown.change(
        fn=lambda interactive_state: gr.update(choices=interactive_state["multi_mask"]["mask_names"]),
        inputs=[interactive_state],
        outputs=[mask_selection_dropdown]
    )
    
    # clear input
    video_input.clear(
        lambda: (
        {
        "user_name": "",
        "video_name": "",
        "origin_images": None,
        "painted_images": None,
        "masks": None,
        "inpaint_masks": None,
        "logits": None,
        "select_frame_number": 0,
        "fps": 30
        },
        {
        "inference_times": 0,
        "negative_click_times" : 0,
        "positive_click_times": 0,
        "mask_save": args.mask_save,
        "multi_mask": {
            "mask_names": [],
            "masks": []
        },
        "track_end_number": 0,
        "resize_ratio": 1,
        "classes": {}
        },
        [[],[]],
        None,
        None,
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False), gr.update(visible=False, value=[]), gr.update(visible=False), \
        gr.update(visible=False), gr.update(visible=False),
        gr.update(visible=False),
        gr.update(value="<p>No classes added yet</p>")
                        
        ),
        [],
        [ 
            video_state,
            interactive_state,
            click_state,
            video_output,
            template_frame,
            tracking_video_predict_button, image_selection_slider , track_pause_number_slider,point_prompt, clear_button_click, 
            Add_mask_button, template_frame, tracking_video_predict_button, video_output, mask_dropdown, remove_mask_button,inpaint_video_predict_button, run_status,
            class_ui,
            class_table
        ],
        queue=False,
        show_progress=False)

    # points clear
    clear_button_click.click(
        fn = clear_click,
        inputs = [video_state, click_state,],
        outputs = [template_frame,click_state, run_status],
    )
    # set example
    gr.Markdown("##  Examples")
    gr.Examples(
        examples=[os.path.join(os.path.dirname(__file__), "./test_sample/", test_sample) for test_sample in ["test-sample8.mp4","test-sample4.mp4", \
                                                                                                             "test-sample2.mp4","test-sample13.mp4"]],
        fn=run_example,
        inputs=[
            video_input
        ],
        outputs=[video_input],
        # cache_examples=True,
    ) 
iface.queue(concurrency_count=1)
iface.launch(debug=True, enable_queue=True, server_port=args.port, server_name="0.0.0.0")
# iface.launch(debug=True, enable_queue=True)