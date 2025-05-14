import gradio as gr
from facial.interface import process_image
from vocal.audio_processor import analyze_audio
from body.main import start_pose_assessment, stop_pose_assessment
from llm_eval.eval import evaluate_stroke_risk


with gr.Blocks(title="StrokeSense: A Multimodal Stroke Assessment Tool") as demo:
    gr.Markdown("# 🧠 StrokeSense\n一个多模态脑卒中初筛系统，包括面部图像分析、语言分析与上肢评估")
    
    with gr.Tab("综合评估"):
        
        gr.Markdown("实时展示三项分析结果，并提供综合判断")
    
        face_display = gr.Textbox(value="尚未完成测评", label="面部分析结果")
        audio_display = gr.Textbox(value="尚未完成测评", label="语言分析结果")
        arm_display = gr.Textbox(value="尚未完成测评", label="上肢评估结果")

        llm_button = gr.Button("综合分析")
        llm_output = gr.Textbox(label="大模型综合评估报告")  
        
        # 综合分析按钮调用大模型
        llm_button.click(
            fn=evaluate_stroke_risk,
            inputs=[face_display, audio_display, arm_display],
            outputs=llm_output
        )

    with gr.Tab("面部图像分析"):
        with gr.Row():
            img_input = gr.Image(type="filepath", label="上传正面面部图像")
            img_output = gr.Image(label="关键点可视化")
        face_result_text = gr.Textbox(label="分析结果")
        img_button = gr.Button("开始分析")
        
        img_button.click(
            fn=process_image,
            inputs=img_input,
            outputs=[img_output, face_result_text]
        ).then(
            fn=lambda x: x,
            inputs=face_result_text,
            outputs=face_display,
        )

    with gr.Tab("语言音频分析"):
        with gr.Row():
            audio_input = gr.Audio(type="filepath", label="上传语音音频")
            audio_output = gr.Textbox(label="音频分析报告")
        audio_button = gr.Button("开始分析")
        
            
        audio_button.click(
            fn=analyze_audio,
            inputs=audio_input,
            outputs=audio_output
        ).then(
            fn=lambda x: x,
            inputs=audio_output,
            outputs=audio_display
        )

    with gr.Tab("上肢运动评估"):
        
        gr.Markdown("点击下方按钮启动或停止摄像头进行NIH上肢动作检测")
        with gr.Row():
            pose_result = gr.Textbox(label="状态")
        with gr.Row():
            start_button = gr.Button("开始评估")
            stop_button = gr.Button("停止评估")
            
        arm_input = gr.Video(format="mp4")
            
        
        start_button.click(
            fn=start_pose_assessment,
            inputs=arm_input,
            outputs=pose_result
        ).then(
            fn=lambda x: x,
            inputs=pose_result, 
            outputs=arm_display,
        )
        
        stop_button.click(fn=stop_pose_assessment, outputs=pose_result)

if __name__ == "__main__":
    demo.launch()