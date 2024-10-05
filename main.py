from flask import Flask, request, render_template, jsonify
from ultralytics.utils import threaded
from werkzeug.utils import secure_filename
import os
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig, pipeline
from PIL import Image
import torch

app = Flask(__name__)

# 设置上传文件夹
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 加载模型和处理器
device = "cpu"
processor = AutoProcessor.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto',
                                          device_map='cpu')
model = AutoModelForCausalLM.from_pretrained('allenai/Molmo-7B-D-0924', trust_remote_code=True, torch_dtype='auto',
                                             device_map='cpu')
translator = pipeline(model='Helsinki-NLP/opus-mt-en-zh', device_map="cpu")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # 获取描述提示，如果未提供则使用默认值
    prompt = request.form.get('prompt', 'What did you see?')

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # 处理图像
        image = Image.open(filepath)
        inputs = processor.process(images=[image], text=prompt)
        inputs = {k: v.to(device).unsqueeze(0) for k, v in inputs.items()}

        output = model.generate_from_batch(
            inputs,
            GenerationConfig(max_new_tokens=512, stop_strings="<|endoftext|>"),
            tokenizer=processor.tokenizer
        )

        generated_tokens = output[0, inputs['input_ids'].size(1):]
        generated_text = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        translated_text = translator(generated_text)[0].get("translation_text")

        return jsonify({
            'description': generated_text,
            'translation': translated_text
        })


if __name__ == '__main__':
    app.run(debug=True,threaded=True,host="0.0.0.0",port=7777)