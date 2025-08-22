# llm_client.py (已加入延时以解决速率限制问题)

import os
import openai
import time  # 1. 导入 time 模块
from dotenv import load_dotenv
from prompt_templates import THINKING_STRATEGIES
from sentence_transformers import SentenceTransformer, util

# 加载 .env 文件中的环境变量
load_dotenv()

class LLMClient:
    """
    一个与大语言模型（LLM）API交互的客户端。
    它封装了API调用，并能执行多种高级思考策略。
    """
    def __init__(self):
        self.api_key = os.getenv("KIM_API_KEY")
        if not self.api_key:
            raise ValueError("未在.env文件中找到KIM_API_KEY，请检查。")
        
        try:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.moonshot.cn/v1",
            )
            self.similarity_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            raise RuntimeError(f"客户端或相似度模型初始化失败: {e}")

    def generate_text(self, prompt, max_tokens=2048):
        """最基础的文本生成函数，用于生成基准答案。"""
        try:
            messages = [{"role": "user", "content": prompt}]
            response = self.client.chat.completions.create(
                model="moonshot-v1-8k",
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            
            # 2. 在每次成功调用API后，强制等待21秒
            print("API call successful. Waiting for 21 seconds to avoid rate limit...")
            time.sleep(21)
            
            return response.choices[0].message.content
        except Exception as e:
            # 3. 将原始错误信息传递出去，方便调试
            raise ConnectionError(f"调用LLM API失败: {e}")

    def _generate_with_template(self, template, **kwargs):
        """内部辅助函数，用于格式化prompt并调用API"""
        prompt = template.format(**kwargs)
        return self.generate_text(prompt)

    def iterative_thinking(self, prompt, strategy_key, max_iterations, convergence_threshold, progress_callback):
        """
        执行所有“迭代式”思考策略 (如自我反思, 对抗式思考)。
        """
        strategy_prompts = THINKING_STRATEGIES[strategy_key]
        critique_template = strategy_prompts["critique"]
        refinement_template = strategy_prompts["refinement"]

        progress_callback("正在生成初始答案...", 0)
        current_answer = self.generate_text(prompt)
        
        history = {"initial": current_answer, "steps": []}

        for i in range(max_iterations):
            progress_callback(f"第 {i+1}/{max_iterations} 轮迭代...", i)
            previous_answer = current_answer

            critique = self._generate_with_template(critique_template, prompt=prompt, current_answer=current_answer)
            refinement = self._generate_with_template(refinement_template, prompt=prompt, current_answer=current_answer, critique=critique)
            current_answer = refinement

            similarity = self.calculate_similarity(current_answer, previous_answer)
            history["steps"].append({"critique": critique, "refined_answer": current_answer, "similarity": similarity})
            
            if similarity >= convergence_threshold:
                progress_callback(f"收敛达成 (相似度 {similarity:.4f})，提前结束。", i + 1)
                break
        else:
             progress_callback(f"达到最大迭代次数 {max_iterations}。", max_iterations)

        return current_answer, history

    def pipeline_thinking(self, prompt, strategy_key, progress_callback):
        """
        执行所有“流水线式”思考策略 (如分步骤深化)。
        """
        strategy_prompts = THINKING_STRATEGIES[strategy_key]
        
        progress_callback("步骤 1/3: 生成概览...", 1)
        overview = self._generate_with_template(strategy_prompts["step1_overview"], prompt=prompt)
        
        progress_callback("步骤 2/3: 深度分析...", 2)
        detailed_analysis = self._generate_with_template(strategy_prompts["step2_analysis"], prompt=prompt, overview=overview)
        
        progress_callback("步骤 3/3: 最终完善...", 3)
        final_answer = self._generate_with_template(strategy_prompts["step3_refine"], prompt=prompt, detailed_analysis=detailed_analysis)
        
        progress_callback("流水线处理完成。", 4)
        
        history = {"概览": overview, "详细分析": detailed_analysis}
        return final_answer, history

    def calculate_similarity(self, text1, text2):
        """计算两个文本的余弦相似度"""
        if not text1 or not text2: return 0.0
        embeddings = self.similarity_model.encode([text1, text2], convert_to_tensor=True)
        return util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()

