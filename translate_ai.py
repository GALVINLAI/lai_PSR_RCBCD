from openai import OpenAI
import os
import nbformat
import re

# 从环境变量中读取 API 密钥
api_key = os.getenv("DEEPSEEK_API_KEY")
if not api_key:
    raise ValueError("Please set the DEEPSEEK_API_KEY environment variable.")

client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def translate_text(text, prompt):
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        stream=False
    )
    translation = response.choices[0].message.content.strip()
    return translation

def contains_chinese(text):
    return re.search(r'[\u4e00-\u9fff]', text) is not None

def deal_python_block(content):
    if content.startswith('```python'):
        content = content[len('```python'):].strip()
    if content.endswith('```'):
        content = content[:-len('```')].strip()
    return content
    
def translate_notebook(input_path, output_path, target_language):
    with open(input_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    total_cells = len(notebook.cells)
    contains_chinese_flag = any(contains_chinese(cell.source) for cell in notebook.cells)
    
    if not contains_chinese_flag:
        print(f"No Chinese characters found in the notebook {input_path}. Skipping translation.")
        return
    
    for idx, cell in enumerate(notebook.cells):
        print(f"Processing cell {idx + 1} of {total_cells} in file {input_path} ({(idx + 1) / total_cells:.2%} complete)")
        if contains_chinese(cell.source):
            if cell.cell_type == 'markdown':
                prompt = f"将以下中文Markdown内容翻译为{target_language}。只返回翻译后的Markdown代码。对于latex数学格式，内联数学内容需要使用$符号，行间数学内容需要用$$符号。"
                cell.source = translate_text(cell.source, prompt)
            elif cell.cell_type == 'code':
                prompt = f"将下面代码脚本中的注释翻译为{target_language}。不要修改代码本身。只返回完全翻译的代码脚本。"
                translated_content = translate_text(cell.source, prompt)
                cell.source = deal_python_block(translated_content)
        else:
            print(f"No Chinese characters found in cell {idx + 1} of file {input_path}. Skipping translation.")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(notebook, f)

    print(f"Translation completed for {input_path}. Output saved to {output_path}")

def translate_file(input_path, output_path, target_language, file_type):
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    if contains_chinese(content):
        if file_type == 'markdown':
            prompt = f"将以下中文Markdown内容翻译为{target_language}。只返回翻译后的Markdown代码。"
        elif file_type == 'python':
            prompt = f"将下面代码脚本中的注释翻译为{target_language}。不要修改代码本身。只返回完全翻译的代码脚本。"
        
        print(f"Processing {file_type} file {input_path}")
        translated_content = translate_text(content, prompt)
        translated_content = deal_python_block(translated_content)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(translated_content)

        print(f"Translation completed for {input_path}. Output saved to {output_path}")
    else:
        print(f"No Chinese characters found in {input_path}. Skipping translation.")

def process_files(base_path, target_language='英文'):
    for root, _, files in os.walk(base_path):
        for file in files:
            file_path = os.path.join(root, file)
            if file in excluded_files:
                print(f"Skipping excluded file {file_path}")
                continue
            if file.endswith('.ipynb'):
                output_path = file_path.replace('.ipynb', '_translated.ipynb')
                translate_notebook(file_path, output_path, target_language)
            elif file.endswith('.py'):
                output_path = file_path.replace('.py', '_translated.py')
                translate_file(file_path, output_path, target_language, 'python')
            elif file.endswith('.md'):
                output_path = file_path.replace('.md', '_translated.md')
                translate_file(file_path, output_path, target_language, 'markdown')

# 示例调用
# base_path = 'lai_playground' # 指定文件夹
# base_path = './'
base_path = 'algo'

# 指定要排除的文件名列表
excluded_files = [
    'example.ipynb',
    'bcd_dev.py',
    'bcd_dev_old.py',
    'tanslate_ai.py',
    'gd.py',
    'rcd.py',
]

process_files(base_path, target_language='英文')
