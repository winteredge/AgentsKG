import os
from openai import OpenAI
from dotenv import load_dotenv

# --- 1. 加载和打印配置 ---
print("正在加载 .env 文件...")
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("API_URL")
MODEL_NAME = os.getenv("MODEL")

# 检查配置
if not all([API_KEY, BASE_URL, MODEL_NAME]):
    print("\n--- 错误 ---")
    print("未能从 .env 文件中完整加载配置。")
    print("请确保 .env 文件在当前目录下，并且包含了以下变量：")
    print("API_KEY, API_URL, MODEL")
    exit()

print("配置加载成功:")
print(f"  - API URL (Base URL): {BASE_URL}")
print(f"  - Model Name:         {MODEL_NAME}")
print("-" * 30)


# --- 2. 准备API客户端和请求数据 ---

# 初始化OpenAI客户端
try:
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    print("-" * 30)
    print("正在尝试连接服务器并获取模型列表...")

    models_response = client.models.list()
    available_models = [model.id for model in models_response]

    if not available_models:
        print("❌ 连接成功，但服务器没有返回任何可用模型！")

    print(f"✅ 连接成功！服务器可用模型: {available_models}")
except Exception as e:
    print(f"\n--- 错误 ---")
    print(f"初始化OpenAI客户端时出错: {e}")
    print("请检查BASE_URL格式是否正确。")
    exit()

# 准备一个最简单的请求
simple_prompt = "你好，请用一句话介绍一下你自己。"
messages = [{"role": "user", "content": simple_prompt}]


# --- 3. 发送请求并打印结果 ---
print(f"正在向模型 '{MODEL_NAME}' 发送请求...")
print(f"请求内容: '{simple_prompt}'")
print("-" * 30)

try:
    # 发起API调用
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,  # 使用一个非零温度以测试随机性
        max_tokens=50,  # 限制输出长度
    )

    # --- 如果成功 ---
    print("\n✅ API调用成功！")
    print("-" * 30)

    # 打印完整的返回对象，以查看其结构
    print("服务器返回的完整对象:")
    print(completion)
    print("-" * 30)

    # 提取并打印出模型的回答
    response_text = completion.choices[0].message.content
    print("提取出的模型回答:")
    print(response_text)
    print("-" * 30)


except Exception as e:
    # --- 如果失败 ---
    print(f"\n❌ API调用失败！")
    print("-" * 30)
    print("捕获到的错误信息:")
    print(e)
    print("-" * 30)
