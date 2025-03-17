import os
import time
import json
import requests
import logging
from typing import Union
from pydantic import BaseModel
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
from langchain.agents import Tool
from langchain_core._api.deprecation import LangChainDeprecationWarning
import warnings

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 忽略LangChain弃用警告
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


# 图片验证模型
class ImageResult(BaseModel):
    url: str
    valid: bool = False
    verified_at: float = None

# 工具函数实现
def validate_image(url: str, max_retries=2) -> bool:
    """验证图片链接有效性（带重试机制）"""
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for _ in range(max_retries):
        try:
            response = requests.head(url, headers=headers, timeout=10, allow_redirects=True)
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                return any(ct in content_type for ct in ['image/jpeg', 'image/png'])
        except Exception:
            time.sleep(0.5)
    return False

def image_scraper(query: str) -> str:
    """增强型图片搜索工具（带自动验证）"""
    try:
        serpapi_key = os.getenv("SERPAPI_KEY", "")
        logging.info(f"开始图片搜索，查询词: {query}")
        response = requests.get(
            "https://serpapi.com/search.json",
            params={
                "q": query,
                "tbm": "isch",
                "api_key": serpapi_key,
                "ijn": "0"
            },
            timeout=20
        )
        response.raise_for_status()
        valid_images = []
        for img in response.json().get("images_results", [])[:5]:
            if url := img.get("original"):
                if validate_image(url):
                    valid_images.append(url)
                    if len(valid_images) >= 2:
                        break
        logging.info(f"图片搜索结果: {valid_images}")
        return "\n".join(valid_images) if valid_images else "未找到有效图片"
    except requests.exceptions.HTTPError as e:
        logging.error(f"图片搜索失败（HTTP错误 {e.response.status_code}）：{e.response.text}")
        return f"图片搜索失败（HTTP错误 {e.response.status_code}）：{e.response.text}"
    except Exception as e:
        logging.error(f"图片服务错误：{str(e)}")
        return f"图片服务错误：{str(e)}"

def web_scraper(keyword: str) -> str:
    """增强型网页搜索工具（使用DuckDuckGo API）"""
    try:
        logging.info(f"开始网页搜索，关键词: {keyword}")
        response = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": keyword, "format": "json", "no_html": 1, "no_redirect": 1},
            timeout=15
        )
        response.raise_for_status()
        results = []
        for item in response.json().get("Results", [])[:3]:
            if url := item.get("FirstURL"):
                results.append(f"{item.get('Text', '')}: {url}")
        logging.info(f"网页搜索结果: {results}")
        return "\n".join(results) if results else "无搜索结果"
    except requests.exceptions.HTTPError as e:
        logging.error(f"网页搜索失败（HTTP错误 {e.response.status_code}）：{e.response.text}")
        return f"网页搜索失败（HTTP错误 {e.response.status_code}）：{e.response.text}"
    except Exception as e:
        logging.error(f"网页搜索失败：{str(e)}")
        return f"网页搜索失败：{str(e)}"

def local_search(keyword: str) -> str:
    """本地知识库搜索（支持中英文混合检索）"""
    try:
        with open("local_articles.json", "r", encoding="utf-8") as f:
            data = json.load(f)
            matches = [
                f"{item['title']}: {item['url']}"
                for item in data
                if keyword.lower() in item.get("content", "").lower()
            ][:3]
            logging.info(f"本地知识库搜索结果: {matches}")
            return "\n".join(matches) if matches else "无本地数据"
    except FileNotFoundError:
        logging.error("本地知识库文件未找到")
        return "本地知识库文件未找到"
    except Exception as e:
        logging.error(f"本地错误：{str(e)}")
        return f"本地错误：{str(e)}"


# 工具集初始化
tools = [
    Tool(name="网页搜索", func=web_scraper, description="网络文章搜索"),
    Tool(name="图片搜索", func=image_scraper, description="验证图片搜索"),
    Tool(name="本地搜索", func=local_search, description="本地知识库检索")
]

# 更新提示模板
react_zh_template = """**请严格遵循格式要求！**

可用工具：{tool_names}

格式说明：
1. 图片链接已自动验证有效性
2. 必须包含至少1个图片结果
3. 最终答案需包含清晰分类
4. 答案全部使用中文

操作步骤：
1. 使用【网页搜索】获取3个相关信息链接
2. 使用【图片搜索】获取2个验证过的图片链接
3. 使用【本地搜索】获取本地知识库匹配结果

最终答案格式要求：
文章推荐（必须包含至少3项）：
1. [标题](链接)
2. [标题](链接)
3. [标题](链接)

图片资源（必须包含至少2项）：
1. ![描述](图片链接)
2. ![描述](图片链接)

本地匹配（必须包含）：
1. [标题](链接)

格式模板：
Question: 需要回答的问题
Thought: 分步中文思考
操作: 工具名称（必须精确匹配 {tool_names}）
操作输入: 工具输入
Observation: 斗兽场介绍: https://example.com/colosseum
...(重复次数小于3)
最终答案: 
文章推荐：
1. [罗马必游景点](https://example.com/top-sights)
图片资源：
1. ![斗兽场夜景](https://example.com/colosseum-night.jpg)
本地匹配：
1. [罗马交通指南](https://local.com/transport)

现在处理：

当前查询：{input}
Thought:"""

# 配置加载
config_list = [
    {
        "model": "deepseek-r1:1.5b",
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "api_type": "openai"
    }
]

# Autogen智能体系统
class InformationRetrievalSystem:
    def __init__(self):
        # 主智能体初始化
        self.main_agent = AssistantAgent(
            name="主智能体",
            system_message=react_zh_template,#"提供相关检索信息，必须包含链接和图片",
            llm_config={"config_list": config_list}
        )

        # 备用分析智能体
        self.backup_agent = AssistantAgent(
            name="备用智能体",
            system_message="""当主回答需要补充时，请执行：
        根据之前结果，进行深度思考，生成一段相关的介绍文章，要求不超过1000字""",
            llm_config={"config_list": config_list}
        )

        # 用户代理配置
        self.proxy = UserProxyAgent(
            name="协调员",
            code_execution_config=False,
            human_input_mode="NEVER"
        )

        # 注册回复处理
        self.proxy.register_reply(
            trigger=self._need_backup_analysis,
            reply_func=self._trigger_backup_agent,
            position=0
        )

        # 初始化群聊
        self.group_chat = GroupChat(
            agents=[self.main_agent, self.backup_agent, self.proxy],
            messages=[],
            max_round=6
        )

        # 聊天历史
        self.chat_history = []

    def _need_backup_analysis(self, msg: Union[dict, str]) -> bool:
        """判断是否需要调用备用智能体"""
        content = msg.get('content', '') if isinstance(msg, dict) else str(msg)
        return 'http' not in content

    def _trigger_backup_agent(self, recipient, messages, sender, config):
        """修正后的备用智能体触发逻辑"""
        original_response = messages[-1].get('content', '')
        question = self.proxy.chat_messages[self.main_agent][-1]['content']

        try:
            # 尝试解析JSON格式内容
            original_data = json.loads(original_response)
            original_text = original_data.get("content", "")
        except:
            original_text = original_response
        # 构造符合要求的消息格式
        valid_message = {
            "content": json.dumps({
                "request_type": "supplementary_analysis",
                "original_question": question,
                "previous_response": original_text
            }),
            "role": "user"  # 必须包含角色信息
        }

        # 发送规范格式消息
        self.proxy.send(
            message=valid_message,  # 使用修正后的消息格式
            recipient=self.backup_agent,
            request_reply=True
        )

        # 获取并组合响应
        backup_response = self.proxy.last_message(self.backup_agent).get('content', '')
        return True, {
            "content": f"【主响应】\n{original_response}\n\n【补充分析】\n{backup_response}",
            "role": "assistant"
        }

    def query(self, question: str) -> str:
        """改进的查询接口"""
        try:
            # 清空历史消息
            self.group_chat.messages = []

            # 创建新的群聊管理器
            manager = GroupChatManager(
                groupchat=self.group_chat,
                llm_config={"config_list": config_list}
            )

            # 发起对话并捕获完整流程
            self.proxy.initiate_chat(
                manager,
                message=question,
                max_turns=6,
                clear_history=False
            )

            # 收集所有响应
            response_chain = []
            for msg in self.group_chat.messages:
                if msg['name'] == '主智能体':
                    response_chain.append(f"=== 主智能体 ===")
                    response_chain.append(msg['content'])
                elif msg['name'] == '备用智能体':
                    response_chain.append(f"\n=== 备用分析 ===")
                    response_chain.append(msg['content'])

            final_response = "\n".join(response_chain[-2:])  # 取最后两组消息

            # 记录日志
            self._log_conversation(question, final_response)

            return final_response
        except Exception as e:
            logging.error(f"查询执行失败: {str(e)}")
            return f"系统暂时不可用，请稍后再试。错误详情：{str(e)}"

    def _log_conversation(self, question: str, response: str):
        """记录对话历史"""
        self.chat_history.append({
            "timestamp": time.time(),
            "question": question,
            "response": response
        })

# 系统初始化与测试
if __name__ == "__main__":
    # 初始化本地知识库示例
    if not os.path.exists("local_articles.json"):
        sample_data = [
            {
                "title": "罗马斗兽场建筑特点",
                "content": "罗马斗兽场的建筑特点包括规模宏大、设计精妙等",
                "url": "https://example.com/colosseum"
            }
        ]
        with open("local_articles.json", "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False)

    # 创建系统实例
    system = InformationRetrievalSystem()

    # 测试用例
    print(f"\n{' 完整流程测试 ':=^40}")
    test_case = "请详细介绍一下罗马"
    print(f"测试问题: {test_case}")

    # 分步执行
    system.proxy.initiate_chat(
        system.main_agent,
        message=test_case,
        max_turns=3
    )
    main_response = system.proxy.last_message(system.main_agent)['content']
    print(f"\n[阶段1 - 主智能体响应]\n{main_response}")

    if 'http' not in main_response:
        print("\n[阶段2 - 触发备用分析]")
        system.proxy.initiate_chat(
            system.backup_agent,
            message=f"主响应缺失链接，请补充分析：{test_case}",
            max_turns=2
        )
        backup_response = system.proxy.last_message(system.backup_agent)['content']
        print(f"\n[阶段2 - 备用智能体响应]\n{backup_response}")

        combined = f"{main_response}\n\n--- 补充分析 ---\n{backup_response}"
        print(f"\n[最终组合响应]\n{combined}")
    else:
        print("\n[无需备用分析]")