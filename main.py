import base64
import io
import json
import re
import threading
import time
from datetime import datetime

import pyautogui
from flask import Flask, request, render_template, jsonify, send_from_directory
from openai import OpenAI
from werkzeug.utils import secure_filename
import os
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, pipeline, \
    MllamaForConditionalGeneration, AutoModel, AutoTokenizer
from PIL import Image
import torch


is_tf  = {
    "llava-llama3":True,
    "allenai/Molmo-7B-D-0924":True,
    "llava:7b-v1.6-vicuna-fp16":True,
    "llava:13b-v1.6-vicuna-q8_0":True,
    "unsloth/Llama-3.2-11B-Vision-Instruct":True,
    "minicpm-v:8b-2.6-q8_0":False,
}

app = Flask(__name__)

screenshot_thread = None
screenshot_interval = 30
is_capturing = False



HISTORY_FILE = 'history.json'
# 设置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载模型和处理器
device = "cpu"
# hf_model = "allenai/MolmoE-1B-0924"
hf_model = "allenai/Molmo-7B-D-0924"
molo_processor = AutoProcessor.from_pretrained(hf_model, trust_remote_code=True, torch_dtype='auto',
                                          device_map='cpu')
molo_model = AutoModelForCausalLM.from_pretrained(hf_model, trust_remote_code=True, torch_dtype='auto',
                                             device_map='cpu')

translator = pipeline(model='Helsinki-NLP/opus-mt-en-zh', device_map="cpu")



def contains_chinese(text):
    """
    检查给定的文本是否包含中文字符。

    :param text: 要检查的字符串
    :return: 如果包含中文字符则返回 True，否则返回 False
    """
    # 匹配中文字符的正则表达式
    pattern = re.compile(r'[\u4e00-\u9fff]+')

    # 如果找到匹配项，则字符串包含中文
    if pattern.search(text):
        return True
    else:
        return False


def result_translated_text(text):
    if contains_chinese(text):
        return text,text
    else:
        return text,translator(text)[0].get("translation_text")


def capture_screen():
    global is_capturing
    while is_capturing:

        screenshot = pyautogui.screenshot()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        screenshot.save(f"{UPLOAD_FOLDER}/{filename}")
        model_type = "minicpm"
        prompt = "Detailed description of what you saw. Please describe the scene you saw and the storage of various items in the scene. Please learn more about the alternate description. :"
        print(f"capture_screen running {UPLOAD_FOLDER}/{filename}")
        generated_text, translated_text, generation_time = _process_image(f"{UPLOAD_FOLDER}/{filename}", prompt, model_type)
        save_history(filename, generated_text, translated_text, model_type, generation_time)
        time.sleep(screenshot_interval)


def save_history(filename, description, translation, model_type,generation_time):
    history = []
    if os.path.exists(f"{UPLOAD_FOLDER}/{HISTORY_FILE}"):
        with open(f"{UPLOAD_FOLDER}/{HISTORY_FILE}", 'r') as f:
            history = json.load(f)

    history.append({
        'filename': filename,
        'description': description,
        'translation': translation,
        'model_type': model_type,
        'timestamp': datetime.now().isoformat(),
        'generation_time': generation_time
    })

    with open(f"{UPLOAD_FOLDER}/{HISTORY_FILE}", 'w') as f:
        json.dump(history, f)


def process_image_molmo(filepath, prompt):
    # 处理图像
    print("process_image_molmo local file path:", filepath)
    image = Image.open(filepath)
    inputs = molo_processor.process(images=[image], text=prompt)
    inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

    output = molo_model.generate_from_batch(
        inputs,
        GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
        tokenizer=molo_processor.tokenizer
    )

    generated_tokens = output[0, inputs['input_ids'].size(1):]
    generated_text = molo_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return result_translated_text(generated_text)



# create image encode function
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def process_mistral_rs(filepath, prompt):
    client = OpenAI(
        base_url='http://localhost:1234/v1/',
        api_key='EMPTY',  # required, but unused
    )
    print("process_mistral_rs local file path:", filepath)
    base64_image = encode_image(filepath)
    print("base64_image:", base64_image)
    completion = client.chat.completions.create(
        model="llama-vision",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                    },
                    {
                        "type": "text",
                        "text": prompt,
                    },
                ],
            },
        ],
        max_tokens=512,
        frequency_penalty=1.0,
        top_p=0.1,
        temperature=0,
    )
    resp = completion.choices[0].message.content
    return result_translated_text(resp)


def process_minicpm(filepath, prompt):
    import ollama
    res = ollama.chat(
        model="minicpm-v:8b-2.6-q8_0",
        # model="llava:13b-v1.6-vicuna-q8_0",
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [filepath]
            }
        ]
    )
    return result_translated_text(res['message']['content'])


def process_meta_llama(filepath, prompt):
    model_id = "unsloth/Llama-3.2-11B-Vision-Instruct"
    model = MllamaForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    image = Image.open(filepath)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=512)

    return result_translated_text(processor.decode(output[0]))


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/start_capture', methods=['POST'])
def start_capture():
    global screenshot_thread, is_capturing
    if not is_capturing:
        is_capturing = True
        screenshot_thread = threading.Thread(target=capture_screen)
        screenshot_thread.start()
        return jsonify({"status": "success", "message": "Screen capture started"})
    return jsonify({"status": "error", "message": "Screen capture already running"})

@app.route('/stop_capture', methods=['POST'])
def stop_capture():
    global is_capturing,screenshot_thread
    if is_capturing:
        is_capturing = False
        return jsonify({"status": "success", "message": "Screen capture stopped"})
    return jsonify({"status": "error", "message": "Screen capture not running"})

def _process_image(filepath, prompt,model_type):
        start_time = time.time()
        if model_type == 'Molo':
            generated_text,translated_text = process_image_molmo(filepath, prompt)
        elif model_type == 'mistral_rs':
            generated_text,translated_text = process_mistral_rs(filepath, prompt)
        elif model_type == 'minicpm':
            generated_text,translated_text = process_minicpm(filepath, prompt)
        elif model_type == 'meta-llama':
            generated_text,translated_text = process_meta_llama(filepath, prompt)
        else:
            return jsonify({'error': 'Invalid model type'})
        end_time = time.time()
        generation_time = end_time - start_time

        return generated_text, translated_text,generation_time

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 获取描述提示，如果未提供则使用默认值
    prompt = request.form.get('prompt', 'What did you see?')
    model_type = request.form.get('model', 'meta_vision_llama')
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        generated_text, translated_text,generation_time =  _process_image(filepath, prompt,model_type)

        save_history(filename, generated_text, translated_text,model_type,generation_time)

        return jsonify({
            'description': generated_text,
            'translation': translated_text,
            'filename': filename,
            'model_type': model_type,
            'generation_time': generation_time
        })


@app.route('/get_history')
def get_history():
    if os.path.exists(f"{UPLOAD_FOLDER}/{HISTORY_FILE}"):
        with open(f"{UPLOAD_FOLDER}/{HISTORY_FILE}", 'r') as f:
            history = json.load(f)
        return jsonify(history)
    return jsonify([])


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True,threaded=True,host="0.0.0.0",port=7777)