# app.py

import streamlit as st
from llm_client import LLMClient
from prompt_templates import THINKING_STRATEGIES

@st.cache_resource
def init_llm_client():
    """
    初始化LLMClient。
    使用 @st.cache_resource 装饰器可以确保在整个应用生命周期中，
    LLMClient只被创建一次，避免重复加载模型和API配置。
    """
    try:
        client = LLMClient()
        return client, None
    except Exception as e:
        # 如果初始化失败（例如，找不到API密钥），则返回错误信息
        return None, str(e)

def main():
    """
    Streamlit应用的主函数。
    """
    # --- 页面基础设置 ---
    st.set_page_config(
        page_title="思考深度对比实验平台", 
        page_icon="💡", 
        layout="wide"
    )
    st.title("💡 AI思考深度对比实验平台")
    st.markdown("通过对比 **单次直接回答** 与 **多种迭代/深化策略**，观察AI如何逐步优化答案。")
    st.markdown("---")

    # --- 初始化客户端 ---
    llm_client, error = init_llm_client()
    if error:
        st.error(f"LLM客户端初始化失败: {error}")
        st.info("请确保你的项目根目录下有 `.env` 文件，并且其中定义了 `KIM_API_KEY`。")
        st.info("同时，请检查相关依赖库是否已安装: `pip install streamlit openai python-dotenv sentence-transformers`")
        return # 初始化失败时，终止应用执行

    # --- 侧边栏配置 ---
    with st.sidebar:
        st.header("⚙️ 实验参数配置")
        
        # 从策略库中提取名称用于下拉菜单显示
        strategy_options = {key: value["name"] for key, value in THINKING_STRATEGIES.items()}
        selected_strategy_key = st.selectbox(
            "选择“反复思考”的策略:",
            options=list(strategy_options.keys()),
            # format_func 用于定义下拉菜单中每个选项的显示文本
            format_func=lambda key: strategy_options[key]
        )

        # 根据选择的策略，决定是否显示“最大反思轮次”滑块
        selected_strategy_info = THINKING_STRATEGIES[selected_strategy_key]
        max_iterations = 0
        if selected_strategy_info.get("type") == "iterative":
            max_iterations = st.slider(
                "最大反思轮次:",
                min_value=1, max_value=5, value=2, step=1,
                help="设置循环思考的最大次数。当答案前后变化不大时，可能会提前“收敛”并停止。"
            )

    # --- 主界面 ---
    prompt = st.text_area(
        "请输入您想让AI思考的问题：", 
        height=100, 
        placeholder="例如：如何才能有效地学习一门新的外语？"
    )

    # 创建左右两个布局列
    col1, col2 = st.columns(2)

    with col1:
        st.header("🚀 单次思考")
        # 为“单次思考”的输出创建一个占位符
        single_pass_output = st.empty()

    with col2:
        st.header(f"🔄 {strategy_options[selected_strategy_key]}")
        # 为“反复思考”的最终答案创建一个占位符
        reflective_output = st.empty()
        # 为“反复思考”的思考过程（st.status）创建一个占位符
        status_placeholder = st.empty()

    # “开始思考”按钮
    if st.button("开始思考", type="primary", use_container_width=True):
        if not prompt:
            st.warning("请输入问题后再开始思考！")
        else:
            # 清空所有输出区域，准备显示新结果
            single_pass_output.empty()
            reflective_output.empty()
            status_placeholder.empty()

            # --- 1. 执行单次思考 ---
            with col1:
                with st.spinner("正在进行单次思考..."):
                    single_response = llm_client.generate_text(prompt, max_tokens=2048)
                    single_pass_output.markdown(single_response)

            # --- 2. 执行反复思考 ---
            with col2:
                spinner_text = f"正在使用“{strategy_options[selected_strategy_key]}”策略进行思考..."
                # 使用 st.status 来包裹并展示思考过程
                with status_placeholder.status(spinner_text, expanded=True) as status:
                    result = {}
                    # 根据策略类型调用不同的后端方法
                    if selected_strategy_info.get("type") == "iterative":
                        result = llm_client.generate_with_reflection(
                            prompt,
                            strategy_name=selected_strategy_key,
                            max_iterations=max_iterations
                        )
                    elif selected_strategy_info.get("type") == "pipeline":
                        result = llm_client.generate_with_deepening(
                            prompt,
                            strategy_name=selected_strategy_key
                        )
                    
                    # 在 status 组件内部，渲染思考历史记录
                    for step in result.get("history", []):
                        st.write(f"**{step['type']}**")
                        st.info(step['content'])
                    
                    # 思考结束后，更新status状态为完成，并默认折叠
                    status.update(label="思考完成!", state="complete", expanded=False)

                # 在主界面上，只渲染最终的答案
                final_answer = result.get("final_answer", "抱歉，未能生成最终答案。")
                reflective_output.markdown(final_answer)

# 当该脚本作为主程序运行时，执行main函数
if __name__ == "__main__":
    main()
