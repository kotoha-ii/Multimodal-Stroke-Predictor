import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_ENDPOINT = os.getenv("API_ENDPOINT") # 替换为完整路径
API_TOKEN = os.getenv("API_TOKEN") # 替换为您的令牌


HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

def evaluate_stroke_risk(face_result: str, audio_result: str, arm_result: str) -> str:
    print("开始LLM测评")
    
    prompt = f"""
    
    你是一位专业的神经科医生，现在你获得了一位患者的以下三个初筛检测结果，请你结合分析，判断该患者是否可能患有脑卒中，并说明理由：

    1. **面部图像分析结果**：
    {face_result}

    2. **语言能力分析结果**：
    {audio_result}

    3. **上肢运动评估结果**：
    {arm_result}

    请基于医学常识和临床经验进行综合判断，使用专业术语描述，最后请输出一个明确的结论，例如：“患者疑似存在中风迹象，建议立即就医。” 或者 “初步未发现明显中风风险”。
    """

    data = {
        "model": "deepseek-r1",  # 根据你平台实际支持的模型调整
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }

    response = requests.post(API_ENDPOINT, headers=HEADERS, json=data)

    if response.status_code == 200:
        response_json = response.json()
        reply = response_json['choices'][0]['message']['content']
        return reply
    else:
        print(f"请求失败，状态码：{response.status_code}")
        print("错误详情：", response.text)
        return "大模型分析失败，请稍后再试。"
