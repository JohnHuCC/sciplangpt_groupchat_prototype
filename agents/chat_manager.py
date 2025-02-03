# chat_manager.py

import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

logger = logging.getLogger(__name__)

class ChatManager:
    """管理多個 Agent 之間的對話和協作"""
    
    def __init__(self):
        self.conversations: Dict[str, List[Dict]] = {}
        self.active_agents: Dict[str, List[str]] = {}
        
    async def initiate_conversation(self, 
                                  conversation_id: str,
                                  agents: List['BaseAgent'],
                                  initial_message: str,
                                  max_rounds: int = 10) -> Dict:
        """
        啟動一個新的對話
        
        Args:
            conversation_id: 對話 ID
            agents: 參與對話的 agent 列表
            initial_message: 初始訊息
            max_rounds: 最大對話輪數
        """
        try:
            # 初始化對話歷史
            self.conversations[conversation_id] = []
            self.active_agents[conversation_id] = [agent.name for agent in agents]
            
            # 註冊 agents 之間的互相認識
            for agent in agents:
                other_agents = {
                    other.name: other for other in agents if other.name != agent.name
                }
                agent.register_available_agents(other_agents)
            
            current_message = initial_message
            conversation_summary = []
            
            for round_num in range(max_rounds):
                # 讓 agents 決定下一個處理者
                next_agent = await self._decide_next_agent(
                    conversation_id,
                    current_message,
                    agents,
                    conversation_summary
                )
                
                if not next_agent:
                    logger.info(f"No agent selected to continue the conversation after {round_num} rounds")
                    break
                    
                # 處理當前訊息
                response = await self._process_agent_response(
                    next_agent,
                    current_message,
                    conversation_summary
                )
                
                # 更新對話歷史
                self.conversations[conversation_id].append({
                    "round": round_num,
                    "agent": next_agent.name,
                    "input": current_message,
                    "output": response["content"],
                    "timestamp": datetime.now().isoformat()
                })
                
                # 更新對話摘要供下一輪使用
                conversation_summary.append({
                    "agent": next_agent.name,
                    "message": response["content"]
                })
                
                # 檢查是否需要繼續對話
                if not await self._should_continue_conversation(
                    response["content"],
                    agents,
                    conversation_summary
                ):
                    logger.info(f"Conversation naturally concluded after {round_num + 1} rounds")
                    break
                    
                current_message = response["content"]
                
            return {
                "conversation_id": conversation_id,
                "history": self.conversations[conversation_id],
                "final_response": current_message
            }
            
        except Exception as e:
            logger.error(f"Error in conversation: {str(e)}")
            raise
            
    async def _decide_next_agent(self,
                                conversation_id: str,
                                current_message: str,
                                agents: List['BaseAgent'],
                                conversation_summary: List[Dict]) -> Optional['BaseAgent']:
        """決定下一個處理訊息的 agent"""
        try:
            # 建構決策提示
            prompt = self._construct_decision_prompt(
                current_message,
                agents,
                conversation_summary
            )
            
            # 讓每個 agent 評估是否適合處理當前訊息
            evaluations = await asyncio.gather(*[
                agent.evaluate_capability(prompt)
                for agent in agents
            ])
            
            # 根據評估結果選擇最適合的 agent
            max_score = -1
            selected_agent = None
            
            for agent, evaluation in zip(agents, evaluations):
                if evaluation["score"] > max_score:
                    max_score = evaluation["score"]
                    selected_agent = agent
                    
            return selected_agent
            
        except Exception as e:
            logger.error(f"Error deciding next agent: {str(e)}")
            return None
            
    def _construct_decision_prompt(self,
                                 current_message: str,
                                 agents: List['BaseAgent'],
                                 conversation_summary: List[Dict]) -> str:
        """建構用於決定下一個 agent 的提示文本"""
        prompt = f"""目前的訊息: {current_message}

可用的 agents:
{chr(10).join([f"- {agent.name}: {agent.description}" for agent in agents])}

對話歷史:
{chr(10).join([f"- {msg['agent']}: {msg['message']}" for msg in conversation_summary])}

請評估你是否適合處理這個訊息，考慮:
1. 訊息內容是否符合你的專長
2. 對話脈絡中是否需要你的專業知識
3. 是否有其他 agent 更適合處理
4. 你能為對話貢獻什麼獨特價值

請提供:
1. 適合度評分 (0-1)
2. 評估理由
"""
        return prompt
        
    async def _process_agent_response(self,
                                    agent: 'BaseAgent',
                                    message: str,
                                    conversation_summary: List[Dict]) -> Dict:
        """處理 agent 的回應"""
        try:
            context = {
                "conversation_history": conversation_summary,
                "available_agents": agent.available_agents
            }
            
            response = await agent.process_query(message, context)
            return response
            
        except Exception as e:
            logger.error(f"Error processing agent response: {str(e)}")
            return {
                "content": f"處理訊息時發生錯誤: {str(e)}",
                "error": str(e)
            }
            
    async def _should_continue_conversation(self,
                                          current_response: str,
                                          agents: List['BaseAgent'],
                                          conversation_summary: List[Dict]) -> bool:
        """決定是否需要繼續對話"""
        # 建構決策提示
        prompt = f"""當前回應: {current_response}

對話歷史:
{chr(10).join([f"- {msg['agent']}: {msg['message']}" for msg in conversation_summary])}

請評估是否需要繼續對話，考慮:
1. 當前回應是否已完整回答問題
2. 是否還需要其他 agent 的補充
3. 繼續對話是否能帶來更多價值

請回答 'yes' 或 'no'"""

        # 讓一個 agent 來決定
        deciding_agent = agents[0]  # 使用第一個 agent 做決定
        decision = await deciding_agent.make_decision(prompt)
        
        return decision.lower() == "yes"