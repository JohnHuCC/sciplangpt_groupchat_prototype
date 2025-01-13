from typing import Dict, List, Optional, Any
import json
import os
from pydantic import BaseModel

class AgentTemplate(BaseModel):
    name: str
    description: str
    type: str
    base_prompt: str
    parameters: Dict[str, Any] = {}
    required_inputs: List[str] = []
    knowledge_config: Optional[Dict[str, Any]] = None

class TemplateManager:
    def __init__(self, templates_dir: str = "templates/agent_templates"):
        self.templates_dir = templates_dir
        os.makedirs(templates_dir, exist_ok=True)
        
    async def load_template(self, template_name: str) -> AgentTemplate:
        """載入指定的模板"""
        template_path = os.path.join(self.templates_dir, f"{template_name}.json")
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return AgentTemplate(**data)
        except Exception as e:
            raise ValueError(f"Error loading template {template_name}: {str(e)}")
            
    async def list_templates(self) -> List[Dict[str, Any]]:
        """列出所有可用的模板"""
        templates = []
        for file_name in os.listdir(self.templates_dir):
            if file_name.endswith('.json'):
                try:
                    template = await self.load_template(file_name[:-5])
                    templates.append(template.model_dump())  # 使用 model_dump() 替代 dict()
                except Exception as e:
                    print(f"Error loading template {file_name}: {str(e)}")
        return templates
    
    async def save_template(self, template_name: str, template_data: Dict[str, Any]) -> AgentTemplate:
        """保存新的模板"""
        template_path = os.path.join(self.templates_dir, f"{template_name}.json")
        template = AgentTemplate(**template_data)
        
        try:
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template.model_dump(), f, ensure_ascii=False, indent=2)
            return template
        except Exception as e:
            raise ValueError(f"Error saving template {template_name}: {str(e)}")
            
    async def delete_template(self, template_name: str) -> bool:
        """刪除指定的模板"""
        template_path = os.path.join(self.templates_dir, f"{template_name}.json")
        try:
            if os.path.exists(template_path):
                os.remove(template_path)
                return True
            return False
        except Exception as e:
            raise ValueError(f"Error deleting template {template_name}: {str(e)}")
            
    async def update_template(self, template_name: str, updates: Dict[str, Any]) -> AgentTemplate:
        """更新現有模板"""
        template_path = os.path.join(self.templates_dir, f"{template_name}.json")
        try:
            if not os.path.exists(template_path):
                raise ValueError(f"Template {template_name} does not exist")
                
            with open(template_path, 'r', encoding='utf-8') as f:
                current_data = json.load(f)
                
            current_data.update(updates)
            template = AgentTemplate(**current_data)
            
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(template.model_dump(), f, ensure_ascii=False, indent=2)
                
            return template
        except Exception as e:
            raise ValueError(f"Error updating template {template_name}: {str(e)}")