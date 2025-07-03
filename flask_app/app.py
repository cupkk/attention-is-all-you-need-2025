import sys
import os
import time
import logging

# 添加项目根目录到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)

from flask import Flask, request, render_template, jsonify, send_file
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    CORS = None
    
from translate_api import translate_german_to_english

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# 如果flask-cors可用则启用跨域支持
if CORS_AVAILABLE and CORS:
    CORS(app)  # 允许跨域请求

@app.route("/", methods=["GET", "POST"])
def index():
    sentence = ""
    translation = None
    error_message = None
    translation_time = None

    if request.method == "POST":
        sentence = request.form.get("sentence", "").strip()
        
        if not sentence:
            error_message = "请输入要翻译的德语句子"
        elif len(sentence) > 500:
            error_message = "输入文本过长，请限制在500字符以内"
        else:
            try:
                start_time = time.time()
                logger.info(f"开始翻译: {sentence}")
                
                # 获取翻译结果
                translation = translate_german_to_english(
                    sentence, 
                    beam_size=10, 
                    alpha=0.3, 
                    visualize=False
                )
                
                translation_time = round((time.time() - start_time) * 1000, 2)  # 毫秒
                logger.info(f"翻译完成: {translation} (耗时: {translation_time}ms)")
                
            except FileNotFoundError as e:
                error_message = "模型文件未找到，请确认模型已正确训练并保存"
                logger.error(f"模型文件错误: {e}")
            except Exception as e:
                error_message = f"翻译失败: {str(e)}"
                logger.error(f"翻译错误: {e}")

    return render_template(
        "index.html", 
        sentence=sentence, 
        translation=translation,
        error_message=error_message,
        translation_time=translation_time
    )

@app.route("/api/translate", methods=["POST"])
def api_translate():
    """REST API 翻译接口"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                'error': '请提供要翻译的文本',
                'code': 'MISSING_TEXT'
            }), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({
                'error': '文本不能为空',
                'code': 'EMPTY_TEXT'
            }), 400
        
        if len(text) > 500:
            return jsonify({
                'error': '文本长度超过限制(500字符)',
                'code': 'TEXT_TOO_LONG'
            }), 400
        
        beam_size = data.get('beam_size', 10)
        alpha = data.get('alpha', 0.3)
        
        start_time = time.time()
        translation = translate_german_to_english(
            text, 
            beam_size=beam_size, 
            alpha=alpha, 
            visualize=False
        )
        translation_time = round((time.time() - start_time) * 1000, 2)
        
        return jsonify({
            'translation': translation,
            'source': text,
            'model': 'transformer',
            'translation_time_ms': translation_time,
            'beam_size': beam_size,
            'alpha': alpha,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        })
        
    except Exception as e:
        logger.error(f"API翻译错误: {e}")
        return jsonify({
            'error': f'翻译失败: {str(e)}',
            'code': 'TRANSLATION_ERROR'
        }), 500

@app.route("/api/health", methods=["GET"])
def health_check():
    """健康检查接口"""
    try:
        # 尝试调用翻译函数检查模型状态
        test_translation = translate_german_to_english("Hallo", visualize=False)
        model_status = "healthy" if test_translation else "error"
    except Exception as e:
        model_status = f"error: {str(e)}"
    
    return jsonify({
        'status': 'ok',
        'model_status': model_status,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route("/attention_report.html")
def attention_report():
    """提供注意力可视化报告"""
    report_path = os.path.join(root_dir, "attention_report.html")
    if os.path.exists(report_path):
        return send_file(report_path)
    else:
        return "<h1>注意力可视化报告暂时不可用</h1><p>请先进行一次翻译以生成可视化报告。</p>", 404

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == "__main__":
    logger.info("启动德语翻译Web应用...")
    app.run(host='0.0.0.0', port=5000, debug=True)