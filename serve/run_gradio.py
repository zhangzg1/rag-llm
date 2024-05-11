import sys

sys.path.append("../")
import gradio as gr
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
from qa_chain.Chat_QA_chain_self import Chat_QA_chain_self
from qa_chain.QA_chain_self import QA_chain_self
from typing import Any
import re
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("zhipu_api_key")

LLM_MODEL_DICT = {
    # "openai": ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"],
    # "wenxin": ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"],
    # "xinhuo": ["Spark-1.5", "Spark-2.0"],
    # "llama": ["Atom-7b", "Llama3-8b"],
    # "zhipuai": ["chatglm_pro", "chatglm_std", "chatglm_lite"]
    "openai": ["gpt-3.5-turbo", "gpt-4"],
    "wenxin": ["ERNIE-Bot"],
    "xinhuo": ["Spark-2.0"],
    "llama": ["Atom-7b", "Llama3-8b"],
    "zhipuai": ["chatglm_pro", "chatglm_std"]
}

LLM_MODEL_LIST = sum(list(LLM_MODEL_DICT.values()), [])
INIT_LLM = "chatglm_std"
EMBEDDING_MODEL_LIST = ['zhipuai', 'openai', 'm3e']
INIT_EMBEDDING_MODEL = "zhipuai"
DEFAULT_DB_PATH = "/home/zhangzg/LLM/rags/database/data/Introduction.md"
DEFAULT_PERSIST_PATH = "/home/zhangzg/LLM/rags/vector_db/ok"
AIGC_AVATAR_PATH = "/home/zhangzg/LLM/rags/figures/datawhale_avatar.png"
DATAWHALE_AVATAR_PATH = "/home/zhangzg/LLM/rags/figures/datawhale_avatar.png"
AIGC_LOGO_PATH = "/home/zhangzg/LLM/rags/figures/aigc_logo.png"
DATAWHALE_LOGO_PATH = "/home/zhangzg/LLM/rags/figures/datawhale_logo.png"


def get_model_by_platform(platform):
    return LLM_MODEL_DICT.get(platform, "")


class Model_center():
    """
    存储问答 Chain 的对象
    - chat_qa_chain_self: 以 (model, embedding_model) 为键存储的带历史记录的问答链。
    - qa_chain_self: 以 (model, embedding_model) 为键存储的不带历史记录的问答链。
    """

    def __init__(self):
        self.chat_qa_chain_self = {}
        self.qa_chain_self = {}

    def chat_qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "chatglm_std",
                                  embedding_model: str = "zhipuai", temperature: float = 0.0, top_k: int = 2,
                                  file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH,
                                  vectordb: Any = None, api_key=api_key):
        """
        调用带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding_model) not in self.chat_qa_chain_self:
                self.chat_qa_chain_self[(model, embedding_model)] = Chat_QA_chain_self(model=model, top_k=top_k,
                                                                                       temperature=temperature,
                                                                                       chat_history=chat_history,
                                                                                       embedding_model=embedding_model,
                                                                                       file_path=file_path,
                                                                                       persist_path=persist_path,
                                                                                       api_key=api_key,
                                                                                       vectordb=vectordb)
            chain = self.chat_qa_chain_self[(model, embedding_model)]
            return "", chain.answer(question=question)
        except Exception as e:
            return e, chat_history

    def qa_chain_self_answer(self, question: str, chat_history: list = [], model: str = "chatglm_std",
                             embedding_model: str = "zhipuai", temperature: float = 0.0, top_k: int = 2,
                             file_path: str = DEFAULT_DB_PATH, persist_path: str = DEFAULT_PERSIST_PATH,
                             vectordb: Any = None, api_key=api_key):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            if (model, embedding_model) not in self.qa_chain_self:
                self.qa_chain_self[(model, embedding_model)] = QA_chain_self(model=model, temperature=temperature,
                                                                             top_k=top_k, api_key=api_key,
                                                                             embedding_model=embedding_model,
                                                                             file_path=file_path,
                                                                             persist_path=persist_path,
                                                                             vectordb=vectordb)
            chain = self.qa_chain_self[(model, embedding_model)]
            chat_history.append(
                (question, chain.answer(question)))
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        if len(self.chat_qa_chain_self) > 0:
            for chain in self.chat_qa_chain_self.values():
                chain.clear_history()


def get_vectordb_info(file_path=DEFAULT_DB_PATH, embedding_model: str = None, persist_path=DEFAULT_PERSIST_PATH):
    if embedding_model == "zhipuai" or embedding_model == "m3e":
        vectordb = get_vectordb(file_path, persist_path, embedding_model)
    return ""


def format_chat_prompt(message, chat_history):
    """
    该函数用于格式化聊天 prompt。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    prompt: 格式化后的 prompt。
    """
    # 初始化一个空字符串，用于存放格式化后的聊天 prompt。
    prompt = ""
    # 遍历聊天历史记录。
    for turn in chat_history:
        # 从聊天记录中提取用户和机器人的消息。
        user_message, bot_message = turn
        # 更新 prompt，加入用户和机器人的消息。
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    # 将当前的用户消息也加入到 prompt中，并预留一个位置给机器人的回复。
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    # 返回格式化后的 prompt。
    return prompt


def respond(message, chat_history, model: str = "chatglm_std", history_len=3, temperature=0.1, api_key=api_key):
    """
    该函数用于生成机器人的回复。

    参数:
    message: 当前的用户消息。
    chat_history: 聊天历史记录。

    返回:
    "": 空字符串表示没有内容需要显示在界面上，可以替换为真正的机器人回复。
    chat_history: 更新后的聊天历史记录
    """
    if message == None or len(message) < 1:
        return "", chat_history
    try:
        # 限制 history 的记忆长度
        chat_history = chat_history[-history_len:] if history_len > 0 else []
        # 调用上面的函数，将用户的消息和聊天历史记录格式化为一个 prompt。
        formatted_prompt = format_chat_prompt(message, chat_history)
        # 调用模型生成回复
        llm = model_to_llm(model=model, temperature=temperature, api_key=api_key)
        bot_message = llm(formatted_prompt)
        # 将bot_message中\n换为<br/>
        bot_message = re.sub(r"\\n", '<br/>', bot_message)
        # 将用户的消息和机器人的回复加入到聊天历史记录中。
        chat_history.append((message, bot_message))
        # 返回一个空字符串和更新后的聊天历史记录（这里的空字符串可以替换为真正的机器人回复，如果需要显示在界面上）。
        return "", chat_history
    except Exception as e:
        return e, chat_history


model_center = Model_center()

block = gr.Blocks()
with block as demo:
    with gr.Row(equal_height=True):
        gr.Image(value=AIGC_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False,
                 container=False)

        with gr.Column(scale=2):
            gr.Markdown("""<h1><center>                                                   RAG应用系统🦜🔗</center></h1>
                <center>LLMs-RAG</center>
                """)

        gr.Image(value=DATAWHALE_LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False,
                 container=False)

    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=400, show_copy_button=True, show_share_button=True,
                                 avatar_images=(AIGC_AVATAR_PATH, DATAWHALE_AVATAR_PATH))

            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt/问题")

            with gr.Row():
                # 创建提交按钮。
                db_with_his_btn = gr.Button("Chat db with history")
                db_wo_his_btn = gr.Button("Chat db without history")
                llm_btn = gr.Button("Chat with llm")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")

        with gr.Column(scale=1):
            file = gr.File(label='请选择知识库目录', file_count='single',
                           file_types=['.txt', '.md', '.docx', '.pdf'])

            with gr.Row():
                init_db = gr.Button("知识库文件向量化")
            model_argument = gr.Accordion("参数配置", open=False)
            with model_argument:
                temperature = gr.Slider(0,
                                        1,
                                        value=0.01,
                                        step=0.01,
                                        label="llm temperature",
                                        interactive=True)

                top_k = gr.Slider(1,
                                  10,
                                  value=3,
                                  step=1,
                                  label="vector db search top k",
                                  interactive=True)

                history_len = gr.Slider(0,
                                        5,
                                        value=3,
                                        step=1,
                                        label="history length",
                                        interactive=True)

            model_select = gr.Accordion("模型选择")
            with model_select:
                model = gr.Dropdown(
                    LLM_MODEL_LIST,
                    label="large language model",
                    value=INIT_LLM,
                    interactive=True)

                embedding_model = gr.Dropdown(EMBEDDING_MODEL_LIST,
                                              label="Embedding model",
                                              value=INIT_EMBEDDING_MODEL)

        """
            1、当点击“知识库向量化”时，会将你选中的文件的内容进行向量化，然后存储到数据库中。也就是存储在 DEFAULT_PERSIST_PATH 这个路径下。
            2、当你进行检索问答时，会将默认的文件（ DEFAULT_DB_PATH 路径）进行向量化，然后也存储在 DEFAULT_PERSIST_PATH 这个路径下，
            此时这个路径下就有了两个文件的向量化数据了。所以每当你放入文件点击“向量化”时就会将新文件的内容向量化并且存储在路径下，然后每次点击
            检索问答时，就会将默认文件向量化然后存储在这个路径下。
            3、单独和大模型聊天时，不会使用数据库中的向量化数据，而是直接使用大模型进行回答。但是和大模型聊天是会结合你的历史记录进行回答的。
            所以可以先清除历史记录，然后再和大模型聊天。
            4、进行带历史的检索聊天时，要先清空历史记录，然后再进行检索聊天。不然无法正常使用。
            5、带历史检索聊天和直接与大模型聊天这两个都是会结合聊天记录的，带历史检索和不带历史检索都会进行一次默认文件的向量化。
        """

        # 设置初始化向量数据库按钮的点击事件。当点击时，调用 get_vectordb 函数，并传入用户的文件和希望使用的 Embedding 模型。
        init_db.click(get_vectordb_info, inputs=[file, embedding_model], outputs=[msg])

        # 设置按钮的点击事件。当点击时，调用上面定义的 chat_qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_with_his_btn.click(model_center.chat_qa_chain_self_answer,
                              inputs=[msg, chatbot, model, embedding_model, temperature, top_k], outputs=[msg, chatbot])

        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer,
                            inputs=[msg, chatbot, model, embedding_model, temperature, top_k], outputs=[msg, chatbot])

        # 设置按钮的点击事件。当点击时，调用上面定义的 respond 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        llm_btn.click(respond, inputs=[msg, chatbot, model, history_len, temperature], outputs=[msg, chatbot])

        # 设置文本框的提交事件（即按下Enter键时）。功能与上面的 llm_btn 按钮点击事件相同。
        msg.submit(respond, inputs=[msg, chatbot, model, history_len, temperature], outputs=[msg, chatbot])

        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)

    gr.Markdown("""提醒：<br>
    1. 使用时请先上传自己的知识文件，不然将会解析项目自带的知识库。
    2. 初始化数据库时间可能较长，请耐心等待。
    3. 使用中如果出现异常，将会在文本输入框进行展示，请不要惊慌。 <br>
    """)
# threads to consume the request
gr.close_all()
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
# demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
