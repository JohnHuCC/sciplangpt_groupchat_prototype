from fastapi import HTTPException
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional
import create_embedding
from agents.base_agent import BaseAgent
from pydantic import BaseModel

class AgentConfig(BaseModel):
    name: str
    description: str
    base_prompt: str
    query_templates: List[str] = []
    type: Optional[str] = None

class Agent(BaseModel):
    name: str
    description: str
    created_at: str
    type: Optional[str] = None

class AgentPoolManager:
    def __init__(self, base_dir: str = "agent_pool"):
        self.base_dir = base_dir
        self.agents_file = os.path.join(base_dir, "agents.json")
        self._initialize_pool()
    
    def _initialize_pool(self):
        """初始化 agent pool"""
        os.makedirs(self.base_dir, exist_ok=True)
        if not os.path.exists(self.agents_file):
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
                
    async def _read_agents_file(self) -> List[Dict]:
        """讀取 agents 文件的輔助方法"""
        try:
            if not os.path.exists(self.agents_file):
                return []
            with open(self.agents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading agents file: {str(e)}"
            )

    async def _write_agents_file(self, agents: List[Dict]):
        """寫入 agents 文件的輔助方法"""
        try:
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump(agents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            raise HTTPException(
                status_code=500, 
                detail=f"Error writing agents file: {str(e)}"
            )

    async def ensure_embeddings(self, docs_dir: str) -> bool:
        """確保 embedding 文件存在"""
        try:
            embedding_dir = os.path.join(docs_dir, "embedding")
            if not os.path.exists(embedding_dir):
                print(f"Generating embeddings for {docs_dir}...")
                await create_embedding.process_files_async(docs_dir)
            return True
        except Exception as e:
            print(f"Error ensuring embeddings: {str(e)}")
            return False
    
    async def create_agent(self, config: AgentConfig, knowledge_files: List[str]) -> Dict:
        """創建新的 agent"""
        try:
            # 創建 agent 專屬目錄
            agent_dir = os.path.join(self.base_dir, config.name)
            if os.path.exists(agent_dir):
                raise HTTPException(status_code=400, detail=f"Agent '{config.name}' already exists")
                
            os.makedirs(agent_dir)
            docs_dir = os.path.join(agent_dir, "docs")
            os.makedirs(docs_dir)
            
            # 複製知識文件
            for file_path in knowledge_files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, docs_dir)
                
            # 生成 embeddings
            if not await self.ensure_embeddings(docs_dir):
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to generate embeddings for agent {config.name}"
                )
            
            # 保存 agent 配置
            agent_config = {
                "name": config.name,
                "description": config.description,
                "created_at": datetime.now().isoformat(),
                "base_prompt": config.base_prompt,
                "query_templates": config.query_templates,
                "type": config.type,
                "knowledge_files": [os.path.basename(f) for f in knowledge_files],
                "docs_dir": docs_dir
            }
            
            config_path = os.path.join(agent_dir, "config.json")
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(agent_config, f, ensure_ascii=False, indent=2)
            
            # 更新 agents 列表
            agents = await self._read_agents_file()
            agents.append(Agent(
                name=config.name,
                description=config.description,
                created_at=agent_config["created_at"],
                type=config.type
            ).dict())
            
            await self._write_agents_file(agents)
            return agent_config
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent(self, name: str) -> Optional[BaseAgent]:
        """獲取指定的 agent 實例"""
        try:
            agent_dir = os.path.join(self.base_dir, name)
            config_path = os.path.join(agent_dir, "config.json")
            
            if not os.path.exists(config_path):
                return None
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            docs_dir = config.get('docs_dir') or os.path.join(agent_dir, "docs")
            
            # 確保 embedding 存在
            if not await self.ensure_embeddings(docs_dir):
                print(f"Warning: Failed to ensure embeddings for agent {name}")
            
            # 根據 agent 類型創建不同的實例
            if config.get("type") == "question_based_research":
                from agents.question_based_agent import QuestionBasedResearchAgent
                return QuestionBasedResearchAgent(
                    name=config["name"],
                    base_prompt=config["base_prompt"],
                    docs_dir=docs_dir,
                    query_templates=config.get("query_templates", [])
                )
            else:
                from agents.dynamic_agent import DynamicAgent
                return DynamicAgent(
                    name=config["name"],
                    base_prompt=config["base_prompt"],
                    docs_dir=docs_dir,
                    query_templates=config.get("query_templates", [])
                )
                
        except Exception as e:
            print(f"Error creating agent instance: {str(e)}")
            return None
    
    async def list_agents(self) -> List[Dict]:
        """列出所有 agents"""
        return await self._read_agents_file()
        
    async def delete_agent(self, name: str) -> bool:
        """刪除指定的 agent"""
        try:
            agent_dir = os.path.join(self.base_dir, name)
            if not os.path.exists(agent_dir):
                return False
                
            # 刪除 agent 目錄及其所有內容
            shutil.rmtree(agent_dir)
            
            # 更新 agents 列表
            agents = await self._read_agents_file()
            agents = [a for a in agents if a["name"] != name]
            await self._write_agents_file(agents)
            
            return True
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")
    
    async def update_agent(self, name: str, updates: Dict) -> Optional[Dict]:
        """更新 agent 配置"""
        try:
            agent_dir = os.path.join(self.base_dir, name)
            config_path = os.path.join(agent_dir, "config.json")
            
            if not os.path.exists(config_path):
                return None
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            config.update(updates)
            
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            if "name" in updates or "description" in updates:
                agents = await self._read_agents_file()
                for agent in agents:
                    if agent["name"] == name:
                        agent.update({
                            "name": updates.get("name", name),
                            "description": updates.get("description", agent["description"])
                        })
                await self._write_agents_file(agents)
                
            return config
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")