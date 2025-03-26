# %%

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List

def collect_python_files(directory: str) -> Dict[str, str]:
    """
    Collect paths and contents of all Python files in the specified directory
    
    Args:
        directory: Directory path to scan
    
    Returns:
        Dictionary containing file paths and their contents
    """
    python_files = {}
    base_dir = Path(directory).name
    
    # Use Path object for directory traversal
    for file_path in Path(directory).rglob("*.py"):
        # Skip __pycache__ directory
        if "__pycache__" in str(file_path):
            continue
            
        try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Convert to relative path and include base directory
            relative_path = str(file_path.relative_to(directory))
            full_path = f"{base_dir}/{relative_path}"
            python_files[full_path] = content
            
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")
            
    return python_files

def group_by_directory(files: Dict[str, str], max_files_per_group: int = 20) -> Dict[str, Dict[str, str]]:
    """
    Group files by their directory structure
    
    Args:
        files: Dictionary of file paths and contents
        max_files_per_group: Maximum number of files per group
    
    Returns:
        Dictionary of grouped files
    """
    # First group by directory
    dir_groups = {}
    for file_path, content in files.items():
        dir_name = str(Path(file_path).parent)
        if dir_name not in dir_groups:
            dir_groups[dir_name] = {}
        dir_groups[dir_name][file_path] = content
    
    # Merge small groups and split large groups
    final_groups = {}
    current_group = {}
    current_group_size = 0
    group_index = 1
    
    for dir_name, dir_files in dir_groups.items():
        # If adding this directory's files would exceed the limit
        if current_group_size + len(dir_files) > max_files_per_group and current_group_size > 0:
            # Save current group and start a new one
            final_groups[f"group_{group_index}"] = current_group
            current_group = {}
            current_group_size = 0
            group_index += 1
        
        # Add files to current group
        current_group.update(dir_files)
        current_group_size += len(dir_files)
    
    # Don't forget to save the last group
    if current_group:
        final_groups[f"group_{group_index}"] = current_group
    
    return final_groups

def save_grouped_json(groups: Dict[str, Dict[str, str]], output_base: str) -> None:
    """
    Save each group to a separate JSON file
    
    Args:
        groups: Grouped files dictionary
        output_base: Base name for output files
    """
    output_base = Path(output_base)
    base_name = output_base.stem
    parent_dir = output_base.parent
    
    for group_name, group_files in groups.items():
        output_file = parent_dir / f"{base_name}_{group_name}.json"
        wrapper = {
            "project_files": {
                "description": f"Python source files collection - {group_name}",
                "base_directory": Path().absolute().name,
                "files": group_files
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(wrapper, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(group_files)} files to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Collect Python files content into JSON for context'
    )
    parser.add_argument(
        '-d', '--directory',
        default='../../../dpti/',
        help='Directory to scan (default: current directory)'
    )
    parser.add_argument(
        '-o', '--output',
        default='build/python_files_context.json',
        help='Output JSON file base name (default: python_files_context.json)'
    )
    parser.add_argument(
        '-n', '--num-files',
        type=int,
        default=20,
        help='Maximum number of files per group (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Collect Python files
    python_files = collect_python_files(args.directory)
    
    # Group files
    groups = group_by_directory(python_files, args.num_files)
    
    # Save grouped files
    save_grouped_json(groups, args.output)
    
    print(f"\nTotal files collected: {len(python_files)}")
    print(f"Split into {len(groups)} groups")

if __name__ == "__main__":
    main()
