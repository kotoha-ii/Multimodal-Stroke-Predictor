import gradio as gr
from facial.interface import process_image
from vocal.core.audio_processor import analyze_audio

# 创建 Gradio 界面
demo = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="filepath", label="上传面部图像"),
    outputs=[
        gr.Image(label="面部关键点可视化"),
        gr.Textbox(label="分析结果")
    ],
    title="面部歪斜与EAR检测",
    description="上传一张正面人脸图像，自动分析嘴角歪斜角度与眼部开合程度（EAR）"
)

if __name__ == "__main__":
    demo.launch()