# MultiModusAutoGen

项目简介
  本系统基于 Autogen 和 LangChain 框架构建，实现多模态信息检索与智能问答功能。通过集成网页搜索、图片搜索和本地知识库，为用户提供结构化的信息服务。
核心功能
  多源搜索：支持网络文章（DuckDuckGo）、验证图片（SerpAPI）、本地知识库混合检索
  智能验证：自动验证图片链接有效性，确保返回资源可用
  结构化输出：强制要求包含文章链接、验证图片和本地匹配结果
  备用分析：当主响应缺失链接时自动触发深度分析

# 快速开始
python
from retrieval_system import InformationRetrievalSystem

# 初始化系统
system = InformationRetrievalSystem()

# 执行查询
result = system.query("请详细介绍一下罗马")
print(result)



工具说明
工具名称	功能描述	                      数据源
网页搜索	网络文章检索	                  DuckDuckGo API
图片搜索	验证图片搜索（自动过滤无效链接）	SerpAPI
本地搜索	本地知识库检索（支持中英文）	    local_articles.json

  ![image](https://github.com/user-attachments/assets/179c4094-d321-4292-a47c-9a94dd677bfb)
