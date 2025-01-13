from fastapi import HTTPException
import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import create_embedding
from agents.base_agent import BaseAgent
from pydantic import BaseModel, Field

# 設置日誌
logger = logging.getLogger(__name__)

class AgentConfig(BaseModel):
    name: str
    description: str
    base_prompt: str
    query_templates: List[str] = Field(default_factory=list)
    type: Optional[str] = None

class Agent(BaseModel):
    name: str
    description: str
    created_at: str
    type: Optional[str] = None

class AgentPoolManager:
    def __init__(self, base_dir: str = "agent_pool"):
        self.base_dir = Path(base_dir)
        self.agents_file = self.base_dir / "agents.json"
        self._initialize_pool()
    
    def _initialize_pool(self) -> None:
        """初始化 agent pool"""
        self.base_dir.mkdir(exist_ok=True)
        if not self.agents_file.exists():
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
                
    async def _read_agents_file(self) -> List[Dict]:
        """讀取 agents 文件"""
        try:
            if not self.agents_file.exists():
                return []
            with open(self.agents_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading agents file: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error reading agents file: {str(e)}"
            )

    async def _write_agents_file(self, agents: List[Dict]) -> None:
        """寫入 agents 文件"""
        try:
            with open(self.agents_file, 'w', encoding='utf-8') as f:
                json.dump(agents, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error writing agents file: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Error writing agents file: {str(e)}"
            )

    async def ensure_embeddings(self, docs_dir: Union[str, Path]) -> bool:
        """確保 embedding 文件存在"""
        try:
            docs_dir = Path(docs_dir)
            embedding_dir = docs_dir / "embedding"
            if not embedding_dir.exists():
                logger.info(f"Generating embeddings for {docs_dir}...")
                await create_embedding.process_files_async(str(docs_dir))
            return True
        except Exception as e:
            logger.error(f"Error ensuring embeddings: {str(e)}")
            return False
    
    async def create_agent(self, config: AgentConfig, knowledge_files: List[str]) -> Dict:
        """創建新的 agent"""
        try:
            logger.info(f"Creating agent: {config.name}")
            
            # 創建 agent 專屬目錄
            agent_dir = self.base_dir / config.name
            if agent_dir.exists():
                raise HTTPException(status_code=400, detail=f"Agent '{config.name}' already exists")
                
            agent_dir.mkdir(parents=True)
            docs_dir = agent_dir / "docs"
            docs_dir.mkdir()
            
            # 複製知識文件
            for file_path in knowledge_files:
                if os.path.exists(file_path):
                    shutil.copy2(file_path, str(docs_dir))
                
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
                "docs_dir": str(docs_dir)
            }
            
            config_path = agent_dir / "config.json"
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
            logger.info(f"Successfully created agent: {config.name}")
            return agent_config
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error creating agent: {str(e)}")
            if agent_dir.exists():
                shutil.rmtree(agent_dir)
            raise HTTPException(status_code=500, detail=str(e))
    
    async def get_agent(self, name: str) -> Optional[Dict[str, Any]]:
        """獲取 agent 配置"""
        try:
            logger.info(f"Getting agent configuration for: {name}")
            agent_dir = self.base_dir / name
            config_path = agent_dir / "config.json"
            
            if not config_path.exists():
                logger.error(f"Config file not found for agent {name}")
                return None
                    
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    logger.info(f"Successfully loaded config for agent {name}")
            except json.JSONDecodeError as e:
                logger.error(f"Error decoding config file for agent {name}: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error reading config file for agent {name}: {str(e)}")
                return None

            # 建立回傳的配置字典，只包含必要且可序列化的資料
            safe_config = {
                "name": config.get("name", name),
                "description": config.get("description", ""),
                "type": config.get("type", "dynamic"),
                "base_prompt": config.get("base_prompt", ""),
                "created_at": config.get("created_at", datetime.now().isoformat()),
                "parameters": {
                    k: v for k, v in config.get("parameters", {}).items()
                    if isinstance(v, (str, int, float, bool, list, dict)) and k != "client"
                },
                "docs_dir": str(config.get("docs_dir", str(agent_dir / "docs"))),
                "knowledge_files": config.get("knowledge_files", [])
            }

            logger.info(f"Successfully prepared configuration for agent {name}")
            return safe_config
            
        except Exception as e:
            logger.error(f"Error getting agent configuration for {name}: {str(e)}")
            return None
        
    # 在 get_agent_instance 中添加配置驗證
    async def get_agent_instance(self, name: str) -> Optional[BaseAgent]:
        try:
            logger.info(f"Getting agent instance: {name}")
            agent_dir = self.base_dir / name
            config_path = agent_dir / "config.json"
            
            if not config_path.exists():
                logger.error(f"Config file not found for agent {name}")
                return None
                    
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Loaded config for agent {name}: {config}")
            
            # 驗證必要的配置欄位
            required_fields = ["name", "base_prompt"]
            missing_fields = [field for field in required_fields if field not in config]
            if missing_fields:
                logger.error(f"Missing required fields in config: {missing_fields}")
                return None
            
            if not config.get("base_prompt"):
                logger.error("base_prompt is empty in config")
                return None

            docs_dir = config.get('docs_dir') or str(agent_dir / "docs")
            
            # 確保 embedding 存在
            if not await self.ensure_embeddings(docs_dir):
                logger.warning(f"Failed to ensure embeddings for agent {name}")
            
            try:
                # 添加更多日誌來追蹤 agent 創建過程
                logger.info(f"Creating agent with config: {json.dumps(config, indent=2)}")
                
                agent_class = None
                if config.get("type") == "question_based_research":
                    logger.info("Creating QuestionBasedResearchAgent")
                    from agents.question_based_agent import QuestionBasedResearchAgent
                    agent_class = QuestionBasedResearchAgent
                else:
                    logger.info("Creating DynamicAgent")
                    from agents.dynamic_agent import DynamicAgent
                    agent_class = DynamicAgent
                
                # 創建 agent 實例
                agent = agent_class(
                    name=config["name"],
                    base_prompt=config["base_prompt"],
                    docs_dir=docs_dir,
                    parameters=config.get("parameters", {})
                )
                
                # 驗證創建的 agent 實例
                if not hasattr(agent, 'process_query'):
                    logger.error("Created agent instance does not have process_query method")
                    return None
                    
                logger.info(f"Successfully created agent instance: {agent}")
                return agent
                
            except ImportError as e:
                logger.error(f"Failed to import agent class: {str(e)}")
                return None
            except Exception as e:
                logger.error(f"Error creating agent instance: {str(e)}")
                logger.exception("Stack trace:")  # 添加堆疊追蹤
                return None
                
        except Exception as e:
            logger.error(f"Error getting agent: {str(e)}")
            logger.exception("Stack trace:")  # 添加堆疊追蹤
            return None
    
    async def list_agents(self) -> List[Dict]:
        """列出所有 agents"""
        try:
            agents = await self._read_agents_file()
            logger.info(f"Listed {len(agents)} agents")
            # 確保回傳的是正確的格式
            formatted_agents = []
            for agent in agents:
                if isinstance(agent, dict):
                    formatted_agents.append({
                        "name": agent.get("name", ""),
                        "description": agent.get("description", ""),
                        "created_at": agent.get("created_at", ""),
                        "type": agent.get("type", "")
                    })
                # 如果是 Agent model 實例
                elif isinstance(agent, Agent):
                    formatted_agents.append({
                        "name": agent.name,
                        "description": agent.description,
                        "created_at": agent.created_at,
                        "type": agent.type
                    })
            return formatted_agents
        except Exception as e:
            logger.error(f"Error listing agents: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error listing agents: {str(e)}"
            )
        
    async def delete_agent(self, name: str) -> bool:
        """刪除指定的 agent"""
        try:
            agent_dir = self.base_dir / name
            if not agent_dir.exists():
                return False
                
            # 刪除 agent 目錄及其所有內容
            shutil.rmtree(agent_dir)
            
            # 更新 agents 列表
            agents = await self._read_agents_file()
            agents = [a for a in agents if a["name"] != name]
            await self._write_agents_file(agents)
            
            logger.info(f"Successfully deleted agent: {name}")
            return True
        except Exception as e:
            logger.error(f"Error deleting agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error deleting agent: {str(e)}")
    
    async def update_agent(self, name: str, updates: Dict) -> Optional[Dict]:
        """更新 agent 配置"""
        try:
            agent_dir = self.base_dir / name
            config_path = agent_dir / "config.json"
            
            if not config_path.exists():
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
                    
            logger.info(f"Successfully updated agent: {name}")
            return config
        except Exception as e:
            logger.error(f"Error updating agent: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error updating agent: {str(e)}")