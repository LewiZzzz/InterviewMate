# 导入所需的库
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

# 创建一个标题和一个副标题
st.title("💬 AI 面试助手")

# 源大模型下载
from modelscope import snapshot_download

model_dir = snapshot_download('IEITYuan/Yuan2-2B-Mars-hf', cache_dir='./')

# 定义模型路径
path = './IEITYuan/Yuan2-2B-Mars-hf'

# 定义模型数据类型
torch_dtype = torch.bfloat16  # A10 GPU支持


# torch_dtype = torch.float16 # P100 GPU支持

# 定义一个函数，用于获取模型和tokenizer
@st.cache_resource
def get_model():
    print("正在创建 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path, add_eos_token=False, add_bos_token=False, eos_token='<eod>')
    tokenizer.add_tokens(
        ['<sep>', '<pad>', '<mask>', '<predict>', '<FIM_SUFFIX>', '<FIM_PREFIX>', '<FIM_MIDDLE>',
         '<commit_before>', '<commit_msg>', '<commit_after>', '<jupyter_start>', '<jupyter_text>',
         '<jupyter_code>', '<jupyter_output>', '<empty_output>'],
        special_tokens=True
    )

    print("正在加载模型...")
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch_dtype, trust_remote_code=True).cuda()

    print("模型加载完毕。")
    return tokenizer, model


# 加载model和tokenizer
tokenizer, model = get_model()

# 使用 radio 组件实现滑块式页面切换
page = st.radio("选择页面", ["模拟面试", "参考面试"])


# 定义Prompt模板
def create_prompt_template(prompt_type, user_input, role=None):
    if prompt_type == "interview_simulation":
        if role:
            return f"你现在是一位面试官，面试求职者应聘的职位是{role}。请只生成一个与此职位相关的面试问题。"
        else:
            return "你现在是一位面试官。请首先询问求职者应聘的岗位。"
    elif prompt_type == "evaluate_answer":
        return f"你现在是一位面试官。求职者的应聘岗位是{role}。请根据以下求职者的回答，评估其回答质量并提供反馈。\n\n求职者的回答: {user_input}\n\n评价与反馈:"
    elif prompt_type == "reference_interview":
        return f"面试问题: {user_input}\n\n参考答案:"


if page == "模拟面试":
    st.header("模拟面试")

    # 初次运行时，session_state中没有"messages"，需要创建一个空列表
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
        st.session_state["interview_role"] = None  # 存储求职者的目标职位
        st.session_state["current_question"] = None  # 存储当前面试问题

    # 如果求职者还没有输入应聘岗位，首先询问岗位
    if st.session_state["interview_role"] is None:
        if prompt := st.chat_input("请输入您应聘的岗位:"):
            st.session_state["interview_role"] = prompt
            st.session_state.messages.append({"role": "user", "content": f"应聘岗位: {prompt}"})
            st.chat_message("user").write(f"应聘岗位: {prompt}")

            # 生成面试官的首个问题
            full_prompt = create_prompt_template("interview_simulation", "", role=st.session_state["interview_role"])
            inputs = tokenizer(full_prompt, return_tensors="pt")["input_ids"].cuda()

            # 调用模型生成问题
            outputs = model.generate(inputs, do_sample=False, max_length=1024)  # 设置解码方式和最大生成长度
            output = tokenizer.decode(outputs[0])
            question = output.split("<sep>")[-1].replace("<eod>", '').strip()

            # 保存并显示问题
            st.session_state["current_question"] = question
            st.session_state.messages.append({"role": "assistant", "content": question})
            st.chat_message("assistant").write(question)
    else:
        # 显示当前问题
        if st.session_state["current_question"]:
            st.chat_message("assistant").write(st.session_state["current_question"])

        # 如果用户在聊天输入框中输入了回答，则执行以下操作
        if answer := st.chat_input("请输入您的回答:"):
            # 将用户的回答添加到session_state中的messages列表中
            st.session_state.messages.append({"role": "user", "content": answer})
            st.chat_message("user").write(answer)

            # 使用 Prompt 模板生成评价输入
            evaluation_prompt = create_prompt_template("evaluate_answer", answer,
                                                       role=st.session_state["interview_role"])
            inputs = tokenizer(evaluation_prompt, return_tensors="pt")["input_ids"].cuda()

            # 调用模型生成评价
            outputs = model.generate(inputs, do_sample=False, max_length=512)  # 设置解码方式和最大生成长度
            output = tokenizer.decode(outputs[0])
            evaluation = output.split("<sep>")[-1].replace("<eod>", '').strip()

            # 显示评价
            st.session_state.messages.append({"role": "assistant", "content": evaluation})
            st.chat_message("assistant").write(evaluation)

            # 重置当前问题以便生成下一个问题
            st.session_state["current_question"] = None

elif page == "参考面试":
    st.header("参考面试")

    # 初次运行时，session_state中没有"reference_messages"，需要创建一个空列表
    if "reference_messages" not in st.session_state:
        st.session_state["reference_messages"] = []

    # 遍历并显示所有聊天记录
    for msg in st.session_state.reference_messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # 如果用户在聊天输入框中输入了问题，则执行以下操作
    if question := st.chat_input("请输入您想咨询的面试问题:"):
        # 将用户的输入添加到session_state中的reference_messages列表中
        st.session_state.reference_messages.append({"role": "user", "content": question})

        # 在聊天界面上显示用户的输入
        st.chat_message("user").write(question)

        # 使用 Prompt 模板生成模型输入
        full_prompt = create_prompt_template("reference_interview", question)
        inputs = tokenizer(full_prompt, return_tensors="pt")["input_ids"].cuda()

        # 调用模型生成参考答案
        outputs = model.generate(inputs, do_sample=False, max_length=512)  # 设置解码方式和最大生成长度
        output = tokenizer.decode(outputs[0])
        reference_answer = output.split("<sep>")[-1].replace("<eod>", '').strip()

        # 删除参考答案之前的内容，只保留核心答案部分
        reference_answer = reference_answer.split("参考答案:")[-1].strip()

        # 将模型的输出添加到session_state中的reference_messages列表中
        st.session_state.reference_messages.append({"role": "assistant", "content": reference_answer})

        # 在聊天界面上显示模型的输出
        st.chat_message("assistant").write(reference_answer)
