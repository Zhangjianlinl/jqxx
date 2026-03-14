"""
OCR工作进程 - 独立进程中运行OCR，避免主程序崩溃
"""
import os
import sys
import json

# 设置环境变量
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_pir_apply_shape_optimization_pass'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.3'
# 设置编码相关环境变量
os.environ['PYTHONIOENCODING'] = 'utf-8'

def get_paddleocr_version():
    """获取paddleocr主版本号"""
    try:
        import paddleocr
        ver = getattr(paddleocr, '__version__', '2.0.0')
        return int(str(ver).split('.')[0])
    except:
        return 2

def run_ocr(image_path):
    """在独立进程中运行OCR，兼容新旧版PaddleOCR API"""
    try:
        from paddleocr import PaddleOCR
        major = get_paddleocr_version()
        texts = []

        if major >= 3:
            # 新版API（PaddleOCR 3.x，Jetson上的新版本）
            ocr = PaddleOCR(
                lang='ch',
                use_textline_orientation=False,
                use_gpu=False
            )
            result = ocr.predict(image_path)
            # 新版返回结构：list of dict，每个dict含 'rec_texts'
            if result and isinstance(result, list):
                for item in result:
                    if isinstance(item, dict):
                        rec_texts = item.get('rec_texts', [])
                        texts.extend([t for t in rec_texts if t and isinstance(t, str)])
        else:
            # 旧版API（PaddleOCR 2.x，Windows上的2.8.1）
            ocr = PaddleOCR(
                lang='ch',
                use_angle_cls=False,
                use_gpu=False
            )
            result = ocr.ocr(image_path)
            if result and isinstance(result, list) and len(result) > 0:
                if result[0] is not None and isinstance(result[0], list):
                    for line in result[0]:
                        if line and isinstance(line, (list, tuple)) and len(line) > 1:
                            if isinstance(line[1], (list, tuple)) and len(line[1]) > 0:
                                text = line[1][0]
                                if text and isinstance(text, str):
                                    texts.append(text)

        return {"success": True, "texts": texts}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    # 设置标准输出编码为UTF-8
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if len(sys.argv) < 2:
        print(json.dumps({"success": False, "error": "No image path provided"}, ensure_ascii=False))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = run_ocr(image_path)
    print(json.dumps(result, ensure_ascii=False, indent=None, separators=(',', ':')))
