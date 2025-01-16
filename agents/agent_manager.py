from typing import Dict, List, Any, Optional
import logging
from datetime import datetime
from agents.base_agent import BaseAgent

logger = logging.getLogger(__name__)

class AgentManager:
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        
    def register_agent(self, agent: BaseAgent):
        """註冊一個 agent"""
        try:
            logger.info(f"Registering agent: {agent.name}")
            self.agents[agent.name] = agent
            # 讓每個 agent 知道其他可用的 agents
            for existing_agent in self.agents.values():
                existing_agent.register_available_agents(self.agents)
            logger.info(f"Successfully registered agent {agent.name}")
        except Exception as e:
            logger.error(f"Error registering agent {agent.name}: {str(e)}")
            raise
    
    def remove_agent(self, agent_name: str):
        """移除一個 agent"""
        try:
            if agent_name in self.agents:
                logger.info(f"Removing agent: {agent_name}")
                del self.agents[agent_name]
                # 更新其他 agents 的可用列表
                for agent in self.agents.values():
                    agent.register_available_agents(self.agents)
                logger.info(f"Successfully removed agent {agent_name}")
            else:
                logger.warning(f"Agent {agent_name} not found for removal")
        except Exception as e:
            logger.error(f"Error removing agent {agent_name}: {str(e)}")
            raise
    
    def get_agent(self, agent_name: str) -> Optional[BaseAgent]:
        """獲取指定的 agent"""
        return self.agents.get(agent_name)
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """列出所有註冊的 agents"""
        return [
            {
                "name": agent.name,
                "description": agent.description,
                "type": agent.__class__.__name__
            }
            for agent in self.agents.values()
        ]
    
    async def process_message(self, message: str, initial_agent: str, context: Optional[Dict] = None) -> Dict:
        """處理消息"""
        try:
            if initial_agent not in self.agents:
                raise ValueError(f"Initial agent {initial_agent} not found")
            
            # 準備處理上下文
            process_context = context or {}
            process_context.update({
                "start_time": datetime.now().isoformat(),
                "message_trail": [],
                "processing_agents": set()  # 使用集合來追蹤已處理的 agents
            })
            
            # 從指定的初始 agent 開始處理
            result = await self.agents[initial_agent].process_query(message, process_context)
            
            # 檢查是否所有 agent 都有參與
            processed_agents = process_context["processing_agents"]
            unprocessed_agents = set(self.agents.keys()) - processed_agents
            
            if unprocessed_agents:
                logger.warning(f"Some agents were not used: {unprocessed_agents}")
                result["unused_agents"] = list(unprocessed_agents)
            
            return {
                "status": "success",
                "result": result,
                "context": process_context,
                "processing_sequence": list(processed_agents)
            }
                
        except Exception as e:
            logger.error(f"Error in agent manager: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "context": context
            }
    
    async def initialize_all(self):
        """初始化所有 agents"""
        try:
            logger.info("Initializing all agents")
            for agent in self.agents.values():
                if hasattr(agent, 'initialize') and callable(getattr(agent, 'initialize')):
                    await agent.initialize()
            logger.info("All agents initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing agents: {str(e)}")
            raise
    
    async def cleanup_all(self):
        """清理所有 agents"""
        try:
            logger.info("Cleaning up all agents")
            for agent in self.agents.values():
                await agent.cleanup()
            logger.info("All agents cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up agents: {str(e)}")
            raise