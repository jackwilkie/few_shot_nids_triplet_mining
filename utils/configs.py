"""
Function to process config files by parsing custom syntax

Created on Wed Oct 11 11:26:55 2023

@author: jack
"""

import ast
import copy
import re
from random import Random
from typing import Any, Dict, Optional
from . import random as random_

PLACEHOLDER_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _resolve_reference(key_path: str, config: Dict[str, Any]) -> Any:
    pointer = config
    for part in key_path.split("."):
        if not isinstance(pointer, dict) or part not in pointer:
            raise KeyError(f"Unable to resolve config reference: {key_path}")
        pointer = pointer[part]
    return pointer


def _resolve_string(value: str, config: Dict[str, Any]) -> Any:
    full_match = PLACEHOLDER_PATTERN.fullmatch(value)
    if full_match is not None:
        return _resolve_reference(full_match.group(1), config)

    def _replace(match: re.Match) -> str:
        replacement = _resolve_reference(match.group(1), config)
        return str(replacement)

    return PLACEHOLDER_PATTERN.sub(_replace, value)


def _resolve_config_values(value: Any, config: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        return _resolve_string(value, config)
    if isinstance(value, dict):
        return {k: _resolve_config_values(v, config) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_config_values(v, config) for v in value]
    return value


def resolve_config(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    changed = True
    while changed:
        next_config = _resolve_config_values(config, config)
        changed = next_config != config
        config = next_config
    return config


def compose_all_configs(config_paths: Dict[str, str]) -> Dict[str, Any]:
    """
    Load and resolve multiple config files with cross-references.

    Parameters
    ----------
    config_paths : Dict[str, str]
        Mapping of config name to path. Example: {
            "dataset": "configs/datasets/lycos.yaml",
            "model": "configs/model.yaml",
            "loss": "configs/loss.yaml",
        }

    Returns
    -------
    Dict[str, Any]
        Dictionary with keys matching config_paths keys, each containing the resolved config.
    """
    configs = {}

    # Load all configs
    for name, path in config_paths.items():
        configs[name] = load_yaml_config(path)

    # Merge into single dict for resolution (allows cross-references)
    merged = copy.deepcopy(configs)

    # Resolve cross-references iteratively
    changed = True
    while changed:
        next_merged = _resolve_config_values(merged, merged)
        changed = next_merged != merged
        merged = next_merged

    # Extract back to separate dicts
    resolved_configs = {}
    for name in configs.keys():
        resolved_configs[name] = merged[name]

    return resolved_configs


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file into a Python dictionary.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.

    Returns
    -------
    Dict[str, Any]
        Parsed configuration dictionary.
    """
    import yaml
    from pathlib import Path

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    with path.open("r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if config is None:
        return {}

    if not isinstance(config, dict):
        raise ValueError(f"Expected YAML config to be a mapping, got {type(config).__name__}.")

    return config


def parse_list(
    start: int,
    stop: int,
    step: int = 1,
    **_
) -> list:
    return [x for x in range(start, stop, step)]

def identity(*args, **kwargs):
    return args, kwargs

def get_command(name):
    commands = dict(
        identity = identity,
        log10_uniform = random_.log10_uniform,
        randchoice = random_.randchoice,
        uniform = random_.uniform,
        randint = random_.randint,
        randpower = random_.randpower,
        make_list = parse_list,
        )
    return commands[name]

def parse_config(
        cnf:dict,
        recursive: bool = True,
        seed:int = None
):
    print(f'config_seed: {seed}')
    cnf = copy.deepcopy(cnf)
    cnf = resolve_config(cnf)
    generator=Random(seed)
    _parse_config(
        cnf,
        recursive = recursive,
        generator = generator
    )

    return cnf

def _parse_config(
        cnf: dict,
        recursive: bool = True,
        generator: Random = None
        ) -> dict:
    
    generator = generator or Random()
    
    for key, value in cnf.items():
        if recursive and isinstance(value, dict):
            _parse_config(value, recursive = recursive, generator=generator)
        
        elif isinstance(value, str):
            cnf[key] = parse_item(value, generator = generator)
            
    return cnf    

def split_args(arg_string: str):
    args = []
    current_arg = ''
    parenthesis_level = 0

    for char in arg_string:
        if char == ',' and parenthesis_level == 0:
            args.append(current_arg.strip())
            current_arg = ''
        else:
            if char == '(':
                parenthesis_level += 1
            elif char == ')':
                parenthesis_level -= 1
            current_arg += char

    if current_arg:
        args.append(current_arg.strip())

    return args


def parse_item(
        string: str,
        generator: Optional[Random] = None
        ):
    
    if isinstance(string, str) and len(string) > 3 and string[:3] == '--:':
        string = string[3:]
    else:
        return string
    
    if not '(' in string and not ')' in string:
        raise ValueError('Invalid comand syntax: missing parenthesis')    
    
    command = get_command(string[:string.find('(')])
    arg_list = string[string.find('(') + 1: string.rfind(')')]
    arg_list = split_args(arg_list)
    args = []
    kwargs = {'generator': generator}
    for arg in arg_list:
        arg = arg.replace(' ', '')
        if not '=' in arg:  # no positional args after kwargs
            arg = parse_item(arg, generator = generator)
            arg = ast.literal_eval(arg) if isinstance(arg, str) else arg
            if len(kwargs) == 1:
                args.append(arg)
            else:
                raise ValueError('Error Positional arg found after keyword arg')
        else: # key word arg
            key = arg[:arg.find('=')]
            arg_string = arg[arg.find('=') + 1:]
            arg_string = parse_item(arg_string, generator=generator)
            arg = ast.literal_eval(arg_string) if isinstance(arg_string, str) else arg_string            
            kwargs[key] = arg

    return command(*args, **kwargs)
