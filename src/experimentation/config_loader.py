"""
Utilitário para carregar e mesclar configurações YAML com override.

Este módulo implementa a estratégia de configuração declarativa proposta,
permitindo carregar base.yaml e aplicar overrides de experimentos específicos.
"""

from pathlib import Path
from typing import Any, Dict
from copy import deepcopy
import yaml


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Faz merge recursivo de dois dicionários.
    
    Valores em `override` substituem valores em `base`.
    Preserva estrutura aninhada.
    
    Args:
        base: Dicionário base
        override: Dicionário com overrides
    
    Returns:
        Dicionário mesclado
    """
    result = deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def load_yaml_file(file_path: Path) -> Dict[str, Any]:
    """Carrega um arquivo YAML."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


