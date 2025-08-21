# llm_client.py

import os
import openai
from dotenv import load_dotenv
from prompt_templates import THINKING_STRATEGIES
from sentence_transformers import SentenceTransformer, util

# 加载 .env 文件中的环境变量
load_dotenv()

class LLMClient:
    """
    一个与大语言模型（LLM）API交互的客户端。
    它封装了API调用、文本生成以及更复杂的思考策略（如反思和深化）。
    """
    def __init__(self):
        """
        初始化客户端，加载API密钥并设置OpenAI客户端。
        同时加载用于文本相似度计算的模型。
        """
        self.api_key = os.getenv("KIM_API_KEY")
        if not self.api_key:
            raise ValueError("未在.env文件中找到KIM_API_KEY，请检查。")
        
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.moonshot.cn/v1",
            )
            # 'all-MiniLM-L6-v2' 是一个轻量级且高效的模型，适合本地计算相似度
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("LLMClient 初始化成功！")
        except Exception as e:
            print(f"LLM 客户端初始化失败: {e}")
            raise

    def _check_convergence(self, answer1: str, answer2: str, threshold: float = 0.98) -> bool:
        """
        通过计算两个文本的语义相似度来检查思考过程是否收敛。
        当新旧答案的相似度超过阈值时，认为答案已经稳定，可以停止迭代。
        """
        # 将文本编码为向量表示
        embeddings = self.similarity_model.encode([answer1, answer2], convert_to_tensor=True)
        # 计算余弦相似度
        cosine_sim = util.pytorch_cos_sim(embeddings[0], embeddings[1])
        print(f"收敛检查：前后两次答案的相似度为 {cosine_sim.item():.4f}")
        return cosine_sim.item() > threshold

    def generate_text(self, prompt: str, max_tokens: int = 2048) -> str:
        """
        调用LLM API生成文本的基础方法。
        增加了 max_tokens 参数以控制单次输出的长度，防止被截断。
        """
        print(f"正在调用Kimi API (max_tokens={max_tokens}): {prompt[:50]}...")
        try:
            response = self.client.chat.completions.create(
                model="moonshot-v1-8k",  # 使用Kimi的8k模型
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,         # 较低的温度使输出更具确定性
                max_tokens=max_tokens    # 控制输出长度
            )
            result = response.choices[0].message.content
            return result.strip()
        except Exception as e:
            print(f"调用 Kimi API 时发生错误: {e}")
            return f"Kimi API 调用失败，错误信息：\n\n{str(e)}"

    def generate_with_reflection(self, prompt: str, strategy_name: str, max_iterations: int = 3) -> dict:
        """
        使用“迭代反思”类策略生成答案（如自我反思、对抗式思考）。
        该方法会返回一个包含最终答案和详细思考历史的字典。
        """
        if strategy_name not in THINKING_STRATEGIES or THINKING_STRATEGIES[strategy_name]['type'] != 'iterative':
            raise ValueError(f"策略 '{strategy_name}' 不是一个有效的迭代策略。")
        
        strategy = THINKING_STRATEGIES[strategy_name]
        history = []
        
        # 生成初版答案，可以给较多的token
        initial_answer = self.generate_text(prompt, max_tokens=2048)
        history.append({"type": "📝 初版答案", "content": initial_answer})
        current_answer = initial_answer

        for i in range(max_iterations):
            print(f"反思过程：正在进行第 {i+1}/{max_iterations} 轮...")
            
            # 生成批判/反思，这部分内容不需要太长，精简即可，节省token
            critique_prompt = strategy["critique"].format(prompt=prompt, current_answer=current_answer)
            critique = self.generate_text(critique_prompt, max_tokens=512) 
            history.append({"type": f"🤔 第 {i+1} 轮反思", "content": critique})

            # 生成优化后的答案，需要完整内容，给足token
            refinement_prompt = strategy["refinement"].format(prompt=prompt, current_answer=current_answer, critique=critique)
            refined_answer = self.generate_text(refinement_prompt, max_tokens=2048)
            
            # 检查答案是否收敛
            if self._check_convergence(current_answer, refined_answer):
                print(f"思考已在第 {i+1} 轮收敛。")
                history.append({"type": f"✨ 修正版答案 #{i+1} (已收敛)", "content": "答案已稳定，生成最终版本。"})
                current_answer = refined_answer
                break

            history.append({"type": f"✨ 修正版答案 #{i+1}", "content": "已根据反思生成新版本。"})
            current_answer = refined_answer
        else: # for-else 循环正常结束时执行
             history.append({"type": "✅ 思考完成", "content": f"已达到最大迭代次数 {max_iterations}。"})

        # 返回包含最终答案和历史记录的字典
        return {"final_answer": current_answer, "history": history}

    def generate_with_deepening(self, prompt: str, strategy_name: str) -> dict:
        """
        使用“流水线式”策略生成答案（如分步骤深化）。
        该方法会按顺序执行策略中的每一步，并返回最终结果和过程历史。
        """
        if strategy_name not in THINKING_STRATEGIES or THINKING_STRATEGIES[strategy_name]['type'] != 'pipeline':
            raise ValueError(f"策略 '{strategy_name}' 不是一个有效的流水线策略。")

        strategy = THINKING_STRATEGIES[strategy_name]
        history = []

        # --- 第1步: 快速概览 ---
        print("深化过程：正在执行第1步 - 快速概览...")
        prompt_step1 = strategy['step1_overview'].format(prompt=prompt)
        overview = self.generate_text(prompt_step1, max_tokens=512) # 概览不需要太长
        history.append({"type": "➡️ 第一步：快速概览", "content": overview})

        # --- 第2步: 详细分析 ---
        print("深化过程：正在执行第2步 - 详细分析...")
        prompt_step2 = strategy['step2_analysis'].format(prompt=prompt, overview=overview)
        detailed_analysis = self.generate_text(prompt_step2, max_tokens=1500) # 分析部分可以长一些
        history.append({"type": "➡️ 第二步：详细分析", "content": detailed_analysis})

        # --- 第3步: 补充完善，生成最终答案 ---
        print("深化过程：正在执行第3步 - 补充完善...")
        prompt_step3 = strategy['step3_refine'].format(prompt=prompt, detailed_analysis=detailed_analysis)
        final_answer = self.generate_text(prompt_step3, max_tokens=2048) # 最终答案需要完整
        history.append({"type": "✅ 第三步：生成最终答案", "content": "已整合所有分析，生成最终版本。"})

        # 返回包含最终答案和历史记录的字典
        return {"final_answer": final_answer, "history": history}

