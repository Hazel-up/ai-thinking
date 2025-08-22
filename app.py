import streamlit as st
from llm_client import LLMClient
from prompt_templates import THINKING_STRATEGIES

# --- 页面基础设置 ---
st.set_page_config(
    page_title="AI高级思考策略对比平台",
    page_icon="🧠",
    layout="wide"
)

# --- 侧边栏：参数配置 ---
with st.sidebar:
    st.header("⚙️ 参数设置")

    # 从prompt_templates动态生成策略选项
    strategy_options = {details["name"]: key for key, details in THINKING_STRATEGIES.items()}
    selected_strategy_name = st.selectbox(
        "选择高级思考策略",
        options=list(strategy_options.keys()),
        key="strategy_name",
        help="选择一个高级思考模型，其结果将与右侧的直接生成结果进行对比。"
    )
    
    selected_strategy_key = strategy_options[selected_strategy_name]
    selected_strategy_type = THINKING_STRATEGIES[selected_strategy_key]["type"]

    if selected_strategy_type == "iterative":
        st.markdown("---")
        st.subheader("迭代参数")
        max_iterations = st.slider("最大迭代次数", 1, 10, 3, 1, help="迭代思考的上限次数。")
        convergence_threshold = st.slider("收敛相似度阈值", 0.8, 1.0, 0.98, 0.01, help="当新旧答案的相似度高于此值时，视为思考收敛，提前停止迭代。")
    else:
        st.markdown("---")
        st.info(f"“{selected_strategy_name}”是一个固定的流水线流程，无须配置迭代参数。")

# --- 主界面 ---
st.title("🧠 AI高级思考策略对比平台")
st.markdown("输入一个问题，平台将同时展示AI的**直接答案**和运用**高级思考策略**后的答案，方便您进行对比分析。")

prompt = st.text_area(
    "请输入您的问题或任务",
    height=150,
    placeholder="例如：如何系统性地学习金融学？请给出一个包含核心领域、学习路径和推荐资源在内的详细计划。"
)

if st.button(f"🚀 开始对比分析", use_container_width=True):
    if not prompt:
        st.warning("请输入您的问题后再开始。")
    else:
        try:
            client = LLMClient()
            col1, col2 = st.columns(2)

            # --- 左侧栏：单次直接生成 (作为基准) ---
            with col1:
                st.header("基准答案 (直接生成)")
                with st.spinner("正在生成基准答案..."):
                    baseline_answer = client.generate_text(prompt)
                    st.info(baseline_answer)

            # --- 右侧栏：高级策略生成 ---
            with col2:
                st.header(f"策略答案 ({selected_strategy_name})")
                
                progress_placeholder = st.empty()
                def streamlit_progress_callback(message, _):
                    progress_placeholder.info(message)

                final_answer, history = None, None

                # 根据策略类型调用不同的客户端方法
                if selected_strategy_type == "iterative":
                    final_answer, history = client.iterative_thinking(
                        prompt, selected_strategy_key, max_iterations, convergence_threshold, streamlit_progress_callback
                    )
                elif selected_strategy_type == "pipeline":
                    final_answer, history = client.pipeline_thinking(
                        prompt, selected_strategy_key, streamlit_progress_callback
                    )
                
                progress_placeholder.empty()
                st.success(final_answer)

                with st.expander("点击查看详细思考过程..."):
                    if selected_strategy_type == "iterative":
                        st.markdown("##### 初始答案")
                        st.text(history["initial"])
                        for i, step in enumerate(history["steps"]):
                            st.markdown(f"---")
                            st.markdown(f"##### 第 {i+1} 轮迭代")
                            st.write("**批判/质疑:**", step["critique"])
                            st.write("**优化后答案:**", step["refined_answer"])
                            st.write(f"**与上一轮相似度:** {step['similarity']:.4f}")
                    
                    elif selected_strategy_type == "pipeline":
                        for step_name, step_content in history.items():
                            st.markdown(f"---")
                            st.markdown(f"##### 步骤：{step_name}")
                            st.text(step_content)

        except Exception as e:
            st.error(f"发生严重错误：{e}")
