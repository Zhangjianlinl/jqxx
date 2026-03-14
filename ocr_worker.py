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
                use_textline_orientation=False
            )
            result = ocr.predict(image_path)
            # 调试：打印原始结构到stderr
            import sys
            print(f"[DEBUG] predict result type: {type(result)}", file=sys.stderr)
            if result is not None:
                try:
                    print(f"[DEBUG] result preview: {str(result)[:300]}", file=sys.stderr)
                except:
                    pass
            # 兼容多种返回结构
            if result is not None:
                # 如果是生成器，转成list
                if hasattr(result, '__iter__') and not isinstance(result, (list, dict)):
                    result = list(result)
                if isinstance(result, list):
                    for item in result:
                        if item is None:
                            continue
                        if isinstance(item, dict):
                            # 标准结构: {'rec_texts': [...], 'rec_scores': [...]}
                            rec = item.get('rec_texts') or item.get('texts') or item.get('text', [])
                            if isinstance(rec, list):
                                texts.extend([t for t in rec if t and isinstance(t, str)])
                        elif isinstance(item, list):
                            # 嵌套list结构
                            for sub in item:
                                if isinstance(sub, dict):
                                    rec = sub.get('rec_texts') or sub.get('texts') or sub.get('text', [])
                                    if isinstance(rec, list):
                                        texts.extend([t for t in rec if t and isinstance(t, str)])
                                elif isinstance(sub, (list, tuple)) and len(sub) > 1:
                                    # 旧式 [[box, (text, score)], ...]
                                    if isinstance(sub[1], (list, tuple)) and len(sub[1]) > 0:
                                        t = sub[1][0]
                                        if t and isinstance(t, str):
                                            texts.append(t)
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
