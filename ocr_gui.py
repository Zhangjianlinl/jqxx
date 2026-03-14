"""
高级OCR日期识别GUI工具
现代化设计，支持图片识别和过期日期检测
"""
import os
# 必须在导入 paddle 之前设置环境变量
os.environ['FLAGS_enable_pir_api'] = '0'
os.environ['FLAGS_pir_apply_shape_optimization_pass'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
# 添加更多稳定性设置
os.environ['FLAGS_allocator_strategy'] = 'auto_growth'
os.environ['FLAGS_fraction_of_gpu_memory_to_use'] = '0.3'
os.environ['FLAGS_cudnn_deterministic'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'  # 限制OpenMP线程数

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import threading
import re
from datetime import datetime
import cv2
import numpy as np
import time
import subprocess
import json
import platform

def get_font(size, weight='normal'):
    """跨平台字体选择，不指定字体名让系统自动回退"""
    if weight != 'normal':
        return ('', size, weight)
    return ('', size)

class ModernOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("智能过期日期识别系统")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f2f5')
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # 设置窗口图标和样式
        self.setup_styles()
        
        # 初始化变量
        self.current_image_path = None
        self.ocr_engine = None
        self.ocr_results = []
        self.warning_days = 30  # 默认30天内过期显示警告
        
        # 摄像头相关变量
        self.camera_mode = False  # False=图片模式, True=摄像头模式
        self.camera = None
        self.camera_running = False
        self.last_recognition_time = 0
        self.recognition_interval = 2.0  # 识别间隔（秒）
        
        # 创建界面
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
        # 初始化OCR引擎（后台加载）
        self.init_ocr_engine()
    
    def setup_styles(self):
        """设置现代化样式"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # 按钮样式
        style.configure('Primary.TButton',
                       font=get_font(11, "bold"),
                       padding=12,
                       background='#1890ff',
                       foreground='white',
                       borderwidth=0)
        
        style.map('Primary.TButton',
                 background=[('active', '#40a9ff'), ('pressed', '#096dd9')])
        
        # 次要按钮样式
        style.configure('Secondary.TButton',
                       font=get_font(10),
                       padding=10,
                       background='#ffffff',
                       foreground='#1890ff',
                       borderwidth=1)
        
        # 标签样式
        style.configure('Title.TLabel',
                       font=get_font(24, "bold"),
                       background='#ffffff',
                       foreground='#1890ff')
        
        style.configure('Subtitle.TLabel',
                       font=get_font(12),
                       background='#ffffff',
                       foreground='#8c8c8c')
    
    def create_header(self):
        """创建顶部标题栏"""
        header_frame = tk.Frame(self.root, bg='#ffffff', height=100)
        header_frame.pack(fill='x', pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # 标题
        title_label = ttk.Label(header_frame, 
                               text="🔍 智能过期日期识别系统",
                               style='Title.TLabel')
        title_label.pack(pady=(20, 5))
        
        # 副标题
        subtitle_label = ttk.Label(header_frame,
                                  text="基于 PaddleOCR 的高精度中文日期识别",
                                  style='Subtitle.TLabel')
        subtitle_label.pack()
    
    def create_main_content(self):
        """创建主要内容区域"""
        main_frame = tk.Frame(self.root, bg='#f0f2f5')
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # 左侧：图片预览区
        self.create_image_panel(main_frame)
        
        # 右侧：识别结果区
        self.create_results_panel(main_frame)
    
    def create_image_panel(self, parent):
        """创建图片预览面板"""
        left_frame = tk.Frame(parent, bg='#ffffff', relief='flat', bd=0)
        left_frame.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # 标题
        title_frame = tk.Frame(left_frame, bg='#ffffff')
        title_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        tk.Label(title_frame, 
                text="📷 图片预览",
                font=get_font(14, "bold"),
                bg='#ffffff',
                fg='#262626').pack(side='left')
        
        # 图片显示区域
        self.image_frame = tk.Frame(left_frame, bg='#fafafa', relief='solid', bd=1)
        self.image_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        self.image_label = tk.Label(self.image_frame,
                                    text="点击下方按钮选择图片\n支持 JPG、PNG、BMP 格式",
                                    font=get_font(12),
                                    bg='#fafafa',
                                    fg='#8c8c8c')
        self.image_label.pack(expand=True)
        
        # 按钮区域
        button_frame = tk.Frame(left_frame, bg='#ffffff')
        button_frame.pack(fill='x', padx=20, pady=(10, 20))
        
        # 模式切换按钮
        self.mode_btn = ttk.Button(button_frame,
                                   text="� 切换到摄像头",
                                   style='Secondary.TButton',
                                   command=self.toggle_mode)
        self.mode_btn.pack(side='left', padx=(0, 10))
        
        self.select_btn = ttk.Button(button_frame,
                                     text="� 选择图片",
                                     style='Primary.TButton',
                                     command=self.select_image)
        self.select_btn.pack(side='left', padx=(0, 10))
        
        self.recognize_btn = ttk.Button(button_frame,
                                       text="🚀 开始识别",
                                       style='Primary.TButton',
                                       command=self.start_recognition,
                                       state='disabled')
        self.recognize_btn.pack(side='left')
        
        # 设置按钮
        self.settings_btn = ttk.Button(button_frame,
                                       text="⚙️ 设置",
                                       style='Secondary.TButton',
                                       command=self.show_settings)
        self.settings_btn.pack(side='left', padx=(10, 0))
    
    def create_results_panel(self, parent):
        """创建识别结果面板"""
        right_frame = tk.Frame(parent, bg='#ffffff', relief='flat', bd=0)
        right_frame.pack(side='right', fill='both', expand=True)
        
        # 标题
        title_frame = tk.Frame(right_frame, bg='#ffffff')
        title_frame.pack(fill='x', padx=20, pady=(20, 10))
        
        tk.Label(title_frame,
                text="📋 识别结果",
                font=get_font(14, "bold"),
                bg='#ffffff',
                fg='#262626').pack(side='left')
        
        # 结果显示区域（带滚动条）
        result_container = tk.Frame(right_frame, bg='#ffffff')
        result_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        scrollbar = tk.Scrollbar(result_container)
        scrollbar.pack(side='right', fill='y')
        
        self.result_text = tk.Text(result_container,
                                   font=('Consolas', 11),
                                   bg='#fafafa',
                                   fg='#262626',
                                   relief='solid',
                                   bd=1,
                                   padx=15,
                                   pady=15,
                                   yscrollcommand=scrollbar.set,
                                   wrap='word')
        self.result_text.pack(fill='both', expand=True)
        scrollbar.config(command=self.result_text.yview)
        
        # 配置文本标签样式
        self.result_text.tag_config('title', font=get_font(12, "bold"), foreground='#1890ff')
        self.result_text.tag_config('success', font=get_font(11), foreground='#52c41a')
        self.result_text.tag_config('warning', font=get_font(11), foreground='#faad14')
        self.result_text.tag_config('error', font=get_font(11), foreground='#ff4d4f')
        self.result_text.tag_config('normal', font=get_font(10), foreground='#595959')
        
        self.show_welcome_message()
    
    def create_footer(self):
        """创建底部状态栏"""
        footer_frame = tk.Frame(self.root, bg='#ffffff', height=50)
        footer_frame.pack(fill='x', side='bottom')
        footer_frame.pack_propagate(False)
        
        self.status_label = tk.Label(footer_frame,
                                     text="就绪",
                                     font=get_font(10),
                                     bg='#ffffff',
                                     fg='#8c8c8c',
                                     anchor='w')
        self.status_label.pack(side='left', padx=20, pady=10)
        
        # 版本信息
        version_label = tk.Label(footer_frame,
                                text="v1.0.0 | Powered by PaddleOCR",
                                font=get_font(9),
                                bg='#ffffff',
                                fg='#bfbfbf')
        version_label.pack(side='right', padx=20)
    
    def show_welcome_message(self):
        """显示欢迎信息"""
        self.result_text.delete('1.0', 'end')
        welcome = """
╔══════════════════════════════════════╗
║     欢迎使用智能过期日期识别系统     ║
╚══════════════════════════════════════╝

📌 功能特点：
  • 高精度中文OCR识别
  • 支持图片识别和摄像头实时识别
  • 自动提取生产日期和过期日期
  • 智能判断产品是否过期
  • 支持多种日期格式

🚀 使用步骤：

  【图片模式】
  1. 点击"选择图片"按钮
  2. 选择包含日期信息的图片
  3. 点击"开始识别"按钮
  4. 查看识别结果

  【摄像头模式】
  1. 点击"切换到摄像头"按钮
  2. 将产品对准摄像头
  3. 点击"开启实时识别"按钮
  4. 系统会自动识别并显示结果

💡 提示：
  • 支持识别药品、食品包装上的日期信息
  • 识别效果受图片/摄像头清晰度影响
  • 可在设置中调整识别间隔和预警天数
        """
        self.result_text.insert('1.0', welcome, 'normal')
        self.result_text.config(state='disabled')
    
    def init_ocr_engine(self):
        """后台初始化OCR引擎"""
        def load_ocr():
            try:
                self.update_status("正在初始化OCR引擎（首次运行可能需要1-2分钟）...")
                
                # 测试OCR工作进程是否可用
                test_result = self.call_ocr_worker("test")
                
                if test_result:
                    self.ocr_engine = "subprocess"  # 标记使用子进程模式
                    self.root.after(0, self.update_status, "✓ OCR引擎初始化完成，就绪")
                    self.root.after(0, messagebox.showinfo, "提示", "OCR引擎初始化完成！\n现在可以开始识别图片了。")
                else:
                    raise Exception("OCR工作进程测试失败")
                
            except Exception as e:
                error_msg = f"OCR引擎初始化失败: {str(e)}"
                self.root.after(0, self.update_status, error_msg)
                self.root.after(0, messagebox.showerror, "错误", f"初始化失败:\n{str(e)}\n\n请检查 PaddleOCR 和 PaddlePaddle 是否正确安装")
        
        thread = threading.Thread(target=load_ocr, daemon=True)
        thread.start()
    
    def call_ocr_worker(self, image_path):
        """调用OCR工作进程"""
        try:
            if image_path == "test":
                # 测试模式，不实际运行OCR
                return True
            
            # 调用独立的Python进程执行OCR
            result = subprocess.run(
                ['python', 'ocr_worker.py', image_path],
                capture_output=True,
                timeout=30
            )
            # 用errors='replace'避免GBK/UTF-8解码失败
            result_stdout = result.stdout.decode('utf-8', errors='replace')
            result_stderr = result.stderr.decode('utf-8', errors='replace')
            
            if result.returncode == 0:
                try:
                    # 找到最后一行JSON输出（跳过日志行）
                    output_lines = result_stdout.strip().splitlines()
                    json_line = None
                    for line in reversed(output_lines):
                        line = line.strip()
                        if line.startswith('{'):
                            json_line = line
                            break
                    
                    if json_line is None:
                        print(f"未找到JSON输出，stdout: {result_stdout[:200]}")
                        return []
                    
                    data = json.loads(json_line)
                    if data.get('success'):
                        return data.get('texts', [])
                    else:
                        print(f"OCR错误: {data.get('error')}")
                        return []
                        
                except json.JSONDecodeError as e:
                    print(f"JSON解析错误: {e}, stdout: {result_stdout[:200]}")
                    return []
            else:
                print(f"进程错误: {result_stderr}")
                return []
        
        except subprocess.TimeoutExpired:
            print("OCR超时")
            return []
        except Exception as e:
            print(f"调用OCR工作进程失败: {e}")
            return []
    
    def select_image(self):
        """选择图片"""
        file_path = filedialog.askopenfilename(
            title="选择图片",
            filetypes=[
                ("图片文件", "*.jpg *.jpeg *.png *.bmp"),
                ("所有文件", "*.*")
            ]
        )
        
        if file_path:
            self.current_image_path = file_path
            self.display_image(file_path)
            self.recognize_btn.config(state='normal')
            import os
            self.update_status(f"已选择: {os.path.basename(file_path)}")
    
    def display_image(self, image_path):
        """显示图片"""
        try:
            # 加载图片
            image = Image.open(image_path)
            
            # 计算缩放比例
            max_width = self.image_frame.winfo_width() - 40
            max_height = self.image_frame.winfo_height() - 40
            
            if max_width <= 0:
                max_width = 500
            if max_height <= 0:
                max_height = 500
            
            # 等比例缩放
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # 显示图片
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo  # 保持引用
            
        except Exception as e:
            messagebox.showerror("错误", f"无法加载图片: {str(e)}")
    
    def start_recognition(self):
        """开始识别（图片模式或摄像头模式）"""
        if self.camera_mode:
            # 摄像头模式：切换实时识别开关
            if not hasattr(self, 'real_time_recognition'):
                self.real_time_recognition = False
            
            if not self.real_time_recognition:
                # 开启实时识别
                if not self.ocr_engine:
                    messagebox.showwarning("警告", "OCR引擎尚未初始化完成，请稍候")
                    return
                
                self.real_time_recognition = True
                self.last_recognition_time = time.time()
                self.recognize_btn.config(text="⏸️ 停止识别")
                self.update_status(f"实时识别已开启（每{self.recognition_interval}秒识别一次）")
                messagebox.showinfo("提示", f"实时识别已开启\n\n将每隔 {self.recognition_interval} 秒自动识别一次\n识别到日期信息时会自动更新结果")
            else:
                # 停止实时识别
                self.real_time_recognition = False
                self.recognize_btn.config(text="▶️ 开启实时识别")
                self.update_status("实时识别已停止")
        else:
            # 图片模式：识别图片
            if not self.current_image_path:
                messagebox.showwarning("警告", "请先选择图片")
                return
            
            if not self.ocr_engine:
                messagebox.showwarning("警告", "OCR引擎尚未初始化完成，请稍候")
                return
            
            # 禁用按钮
            self.recognize_btn.config(state='disabled')
            self.select_btn.config(state='disabled')
            
            # 直接在主线程中执行识别（避免多线程问题）
            self.root.after(10, self.perform_recognition_safe)
    
    def perform_recognition_safe(self):
        """安全的OCR识别（使用子进程）"""
        try:
            self.update_status("正在识别中...")
            
            # 使用子进程调用OCR
            texts = self.call_ocr_worker(self.current_image_path)
            
            if texts is None:
                texts = []
            
            # 提取日期
            dates = self.extract_dates(texts)
            
            # 显示结果
            self.display_results(texts, dates)
            
        except Exception as e:
            error_detail = f"识别失败: {str(e)}\n\n详细信息:\n{type(e).__name__}"
            messagebox.showerror("错误", error_detail)
            import traceback
            traceback.print_exc()
        finally:
            self.enable_buttons()
    
    def perform_recognition(self):
        """执行OCR识别 - 已废弃，保留以防兼容性问题"""
        pass
    
    def extract_dates(self, texts):
        """从文字中提取日期"""
        dates = []
        patterns = [
            # 带标签的日期（同行）
            (r'生产日期[：:]\s*(\d{4})[\.\-/年](\d{1,2})[\.\-/月](\d{1,2})日?', '生产日期'),
            (r'有效期至[：:]\s*(\d{4})[\.\-/年](\d{1,2})[\.\-/月](\d{1,2})日?', '过期日期'),
            (r'有效期[：:]\s*(\d{4})[\.\-/年](\d{1,2})[\.\-/月](\d{1,2})日?', '过期日期'),
            (r'保质期至[：:]\s*(\d{4})[\.\-/年](\d{1,2})[\.\-/月](\d{1,2})日?', '过期日期'),
            (r'保质期[：:]\s*(\d{4})[\.\-/年](\d{1,2})[\.\-/月](\d{1,2})日?', '过期日期'),
            (r'生产日期[：:]\s*(\d{8})', '生产日期'),
            (r'有效期[：:]\s*(\d{8})', '过期日期'),
            (r'保质期[：:]\s*(\d{8})', '过期日期'),
            (r'生产日期[：:]\s*(\d{6})', '生产日期'),   # YYYYMM格式
            (r'有效期[：:]\s*(\d{6})', '过期日期'),
            (r'保质期[：:]\s*(\d{6})', '过期日期'),
            (r'EXP[：:]\s*(\d{4})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', '过期日期'),
            (r'MFG[：:]\s*(\d{4})[\.\-/](\d{1,2})[\.\-/](\d{1,2})', '生产日期'),
            # 8位紧凑格式 YYYYMMDD
            (r'(?<![0-9])(\d{8})(?![0-9])', '日期'),
            # 6位紧凑格式 YYYYMM
            (r'(?<![0-9])(\d{6})(?![0-9])', '日期'),
            # 标准格式
            (r'(\d{4})[\年\-/](\d{1,2})[\月\-/](\d{1,2})日?', '日期'),
        ]
        
        found_dates = []
        
        # 构建跨行上下文：检查相邻文本行的关键词关联
        # 例如 "【有效期】至" 在第i行，"202509" 在第i+1行
        context_map = {}  # index -> 推断的日期类型
        for i, text in enumerate(texts):
            text_lower = text.lower()
            if '有效期' in text or '保质期' in text or 'exp' in text_lower or '过期' in text:
                # 下一行可能是日期
                if i + 1 < len(texts):
                    context_map[i + 1] = '过期日期'
            if '生产日期' in text or '生产' in text or 'mfg' in text_lower:
                if i + 1 < len(texts):
                    context_map[i + 1] = '生产日期'
        
        for idx, text in enumerate(texts):
            # 检查文本中是否包含关键词来判断日期类型
            text_lower = text.lower()
            has_production = '生产' in text or 'mfg' in text_lower
            has_expiry = '有效期' in text or '保质期' in text or 'exp' in text_lower or '过期' in text
            # 跨行上下文推断
            context_type = context_map.get(idx)
            
            for pattern, date_type in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    try:
                        # 处理紧凑格式（8位YYYYMMDD 或 6位YYYYMM）
                        if len(match.groups()) == 1:
                            date_str = match.group(1)
                            if len(date_str) == 8:
                                year = int(date_str[0:4])
                                month = int(date_str[4:6])
                                day = int(date_str[6:8])
                            elif len(date_str) == 6:
                                year = int(date_str[0:4])
                                month = int(date_str[4:6])
                                import calendar
                                day = calendar.monthrange(year, month)[1]
                            else:
                                continue
                        else:
                            year = int(match.group(1))
                            month = int(match.group(2))
                            day = int(match.group(3))
                        if year < 2000 or year > 2100:
                            continue
                        if month < 1 or month > 12:
                            continue
                        if day < 1 or day > 31:
                            continue
                        
                        date_obj = datetime(year, month, day)
                        
                        # 智能判断日期类型
                        final_date_type = date_type
                        if date_type == '日期':
                            # 优先使用跨行上下文推断
                            if context_type:
                                final_date_type = context_type
                            elif has_expiry:
                                final_date_type = '过期日期'
                            elif has_production:
                                final_date_type = '生产日期'
                            else:
                                # 根据日期与当前时间的关系判断
                                if date_obj > datetime.now():
                                    final_date_type = '过期日期'
                                else:
                                    final_date_type = '生产日期'
                        
                        # 检查过期状态（对所有可能的过期日期类型都检查）
                        status = None
                        if final_date_type in ['过期日期', '保质期至', '有效期至', '有效期']:
                            status = self.check_expiry(date_obj)
                        
                        found_dates.append({
                            'type': final_date_type,
                            'date': date_obj.strftime('%Y年%m月%d日'),
                            'original': text,
                            'status': status,
                            'date_obj': date_obj
                        })
                    except:
                        pass
        
        # 去重（相同日期只保留一个）
        unique_dates = []
        seen_dates = set()
        for date_info in found_dates:
            date_key = (date_info['date'], date_info['type'])
            if date_key not in seen_dates:
                seen_dates.add(date_key)
                unique_dates.append(date_info)
        
        return unique_dates
    
    def check_expiry(self, date_obj):
        """检查是否过期"""
        today = datetime.now()
        days_diff = (date_obj - today).days
        
        if days_diff < 0:
            return ('error', f"已过期 {abs(days_diff)} 天")
        elif days_diff == 0:
            return ('warning', "今天到期")
        elif days_diff <= self.warning_days:
            return ('warning', f"即将过期（还有 {days_diff} 天）")
        else:
            return ('success', f"未过期（还有 {days_diff} 天）")
    
    def show_settings(self):
        """显示设置对话框"""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("设置")
        settings_window.geometry("400x350")
        settings_window.configure(bg='#ffffff')
        settings_window.resizable(False, False)
        
        # 居中显示
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        # 标题
        title_label = tk.Label(settings_window,
                              text="⚙️ 系统设置",
                              font=get_font(14, "bold"),
                              bg='#ffffff',
                              fg='#1890ff')
        title_label.pack(pady=20)
        
        # 设置框架
        frame = tk.Frame(settings_window, bg='#ffffff')
        frame.pack(pady=10, padx=30, fill='both', expand=True)
        
        # 预警天数设置
        label1 = tk.Label(frame,
                        text="即将过期预警天数：",
                        font=get_font(11),
                        bg='#ffffff',
                        fg='#262626')
        label1.grid(row=0, column=0, sticky='w', pady=10)
        
        days_var = tk.IntVar(value=self.warning_days)
        days_spinbox = tk.Spinbox(frame,
                                  from_=1,
                                  to=365,
                                  textvariable=days_var,
                                  font=get_font(11),
                                  width=10)
        days_spinbox.grid(row=0, column=1, padx=10, pady=10)
        
        days_label = tk.Label(frame,
                             text="天",
                             font=get_font(11),
                             bg='#ffffff',
                             fg='#262626')
        days_label.grid(row=0, column=2, sticky='w', pady=10)
        
        # 说明文字
        info_label1 = tk.Label(frame,
                             text=f"当产品距离过期日期少于设定天数时\n将显示黄色警告提示",
                             font=get_font(9),
                             bg='#ffffff',
                             fg='#8c8c8c',
                             justify='left')
        info_label1.grid(row=1, column=0, columnspan=3, sticky='w', pady=(0, 20))
        
        # 识别间隔设置
        label2 = tk.Label(frame,
                        text="实时识别间隔：",
                        font=get_font(11),
                        bg='#ffffff',
                        fg='#262626')
        label2.grid(row=2, column=0, sticky='w', pady=10)
        
        interval_var = tk.DoubleVar(value=self.recognition_interval)
        interval_spinbox = tk.Spinbox(frame,
                                     from_=1.0,
                                     to=10.0,
                                     increment=0.5,
                                     textvariable=interval_var,
                                     font=get_font(11),
                                     width=10)
        interval_spinbox.grid(row=2, column=1, padx=10, pady=10)
        
        interval_label = tk.Label(frame,
                                 text="秒",
                                 font=get_font(11),
                                 bg='#ffffff',
                                 fg='#262626')
        interval_label.grid(row=2, column=2, sticky='w', pady=10)
        
        # 说明文字
        info_label2 = tk.Label(frame,
                             text=f"摄像头模式下每隔设定时间自动识别\n间隔越短识别越频繁，但更耗性能",
                             font=get_font(9),
                             bg='#ffffff',
                             fg='#8c8c8c',
                             justify='left')
        info_label2.grid(row=3, column=0, columnspan=3, sticky='w', pady=(0, 10))
        
        # 按钮框架
        button_frame = tk.Frame(settings_window, bg='#ffffff')
        button_frame.pack(pady=20)
        
        def save_settings():
            self.warning_days = days_var.get()
            self.recognition_interval = interval_var.get()
            messagebox.showinfo("提示", f"设置已保存！\n\n预警天数: {self.warning_days} 天\n识别间隔: {self.recognition_interval} 秒")
            settings_window.destroy()
        
        def cancel_settings():
            settings_window.destroy()
        
        save_btn = ttk.Button(button_frame,
                             text="保存",
                             style='Primary.TButton',
                             command=save_settings)
        save_btn.pack(side='left', padx=5)
        
        cancel_btn = ttk.Button(button_frame,
                               text="取消",
                               style='Secondary.TButton',
                               command=cancel_settings)
        cancel_btn.pack(side='left', padx=5)
    
    def display_results(self, texts, dates):
        """显示识别结果"""
        self.result_text.config(state='normal')
        self.result_text.delete('1.0', 'end')
        
        # 标题
        self.result_text.insert('end', "╔══════════════════════════════════════╗\n", 'title')
        self.result_text.insert('end', "║           识别结果报告               ║\n", 'title')
        self.result_text.insert('end', "╚══════════════════════════════════════╝\n\n", 'title')
        
        # 识别到的所有文字
        self.result_text.insert('end', "📝 识别到的文字：\n", 'title')
        self.result_text.insert('end', "─" * 40 + "\n", 'normal')
        for i, text in enumerate(texts, 1):
            self.result_text.insert('end', f"{i}. {text}\n", 'normal')
        
        self.result_text.insert('end', "\n")
        
        # 日期信息和过期状态分析
        expired_count = 0
        warning_count = 0
        safe_count = 0
        
        if dates:
            self.result_text.insert('end', "📅 提取的日期信息：\n", 'title')
            self.result_text.insert('end', "─" * 40 + "\n", 'normal')
            
            for i, date_info in enumerate(dates, 1):
                self.result_text.insert('end', f"\n【日期 {i}】\n", 'title')
                self.result_text.insert('end', f"  类型: {date_info['type']}\n", 'normal')
                self.result_text.insert('end', f"  日期: {date_info['date']}\n", 'normal')
                self.result_text.insert('end', f"  原文: {date_info['original']}\n", 'normal')
                
                if date_info['status']:
                    tag, status_text = date_info['status']
                    self.result_text.insert('end', f"  状态: {status_text}\n", tag)
                    
                    # 统计过期状态
                    if tag == 'error':
                        expired_count += 1
                    elif tag == 'warning':
                        warning_count += 1
                    elif tag == 'success':
                        safe_count += 1
            
            # 显示汇总预警
            self.result_text.insert('end', "\n" + "═" * 40 + "\n", 'normal')
            self.result_text.insert('end', "⚠️  过期状态汇总：\n", 'title')
            self.result_text.insert('end', "─" * 40 + "\n", 'normal')
            
            if expired_count > 0:
                self.result_text.insert('end', f"❌ 已过期: {expired_count} 个\n", 'error')
            if warning_count > 0:
                self.result_text.insert('end', f"⚠️  即将过期: {warning_count} 个\n", 'warning')
            if safe_count > 0:
                self.result_text.insert('end', f"✅ 未过期: {safe_count} 个\n", 'success')
            
            # 弹窗预警
            self.show_expiry_alert(expired_count, warning_count, safe_count)
            
        else:
            self.result_text.insert('end', "❌ 未找到日期信息\n", 'error')
            self.result_text.insert('end', "\n💡 提示：请确保图片中包含清晰的日期信息\n", 'normal')
        
        self.result_text.insert('end', "\n" + "─" * 40 + "\n", 'normal')
        self.result_text.insert('end', f"识别完成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n", 'normal')
        
        self.result_text.config(state='disabled')
        self.update_status("识别完成")
    
    def show_expiry_alert(self, expired_count, warning_count, safe_count):
        """显示过期预警弹窗"""
        total_dates = expired_count + warning_count + safe_count
        
        if total_dates == 0:
            # 找到了日期但没有过期状态（可能是生产日期等）
            messagebox.showinfo(
                "ℹ️ 提示",
                "已识别到日期信息，但未找到过期日期。\n\n"
                "检测到的可能是生产日期或其他日期类型。\n\n"
                "详细信息请查看识别结果。"
            )
        elif expired_count > 0:
            # 严重警告：已过期
            messagebox.showerror(
                "⚠️ 过期警告",
                f"检测到 {expired_count} 个已过期日期！\n\n"
                f"请立即检查产品，避免使用过期物品。\n\n"
                f"详细信息请查看识别结果。"
            )
        elif warning_count > 0:
            # 提醒：即将过期
            messagebox.showwarning(
                "⏰ 过期提醒",
                f"检测到 {warning_count} 个即将过期的日期！\n\n"
                f"建议尽快使用或处理相关产品。\n\n"
                f"详细信息请查看识别结果。"
            )
        elif safe_count > 0:
            # 正常：未过期
            messagebox.showinfo(
                "✅ 检测完成",
                f"所有日期均未过期！\n\n"
                f"共检测到 {safe_count} 个有效日期。\n\n"
                f"产品在有效期内，可以安全使用。"
            )
    
    def enable_buttons(self):
        """启用按钮"""
        self.recognize_btn.config(state='normal')
        self.select_btn.config(state='normal')
    
    def on_closing(self):
        """窗口关闭事件处理"""
        # 停止实时识别
        if hasattr(self, 'real_time_recognition'):
            self.real_time_recognition = False
        
        # 停止摄像头
        if self.camera is not None:
            self.stop_camera()
        
        self.root.destroy()
    
    def update_status(self, message):
        """更新状态栏"""
        self.status_label.config(text=message)
    
    def toggle_mode(self):
        """切换图片/摄像头模式"""
        if self.camera_mode:
            # 切换到图片模式
            self.stop_camera()
            self.camera_mode = False
            self.mode_btn.config(text="📹 切换到摄像头")
            self.select_btn.config(state='normal')
            self.recognize_btn.config(text="🚀 开始识别", state='disabled')
            self.image_label.config(text="点击下方按钮选择图片\n支持 JPG、PNG、BMP 格式")
            self.update_status("已切换到图片模式")
        else:
            # 切换到摄像头模式
            self.camera_mode = True
            self.mode_btn.config(text="📷 切换到图片")
            self.select_btn.config(state='disabled')
            self.recognize_btn.config(text="▶️ 开启实时识别", state='normal')
            self.start_camera()
            self.update_status("已切换到摄像头模式")
    
    def start_camera(self):
        """启动摄像头"""
        if self.camera is None:
            try:
                self.camera = cv2.VideoCapture(0)
                if not self.camera.isOpened():
                    messagebox.showerror("错误", "无法打开摄像头\n请检查摄像头是否连接正常")
                    self.camera = None
                    return
                
                # 设置摄像头分辨率
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                self.camera_running = True
                self.update_camera_feed()
                self.update_status("摄像头已启动")
            except Exception as e:
                messagebox.showerror("错误", f"启动摄像头失败: {str(e)}")
                self.camera = None
    
    def stop_camera(self):
        """停止摄像头"""
        self.camera_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.update_status("摄像头已停止")
    
    def update_camera_feed(self):
        """更新摄像头画面"""
        if not self.camera_running or self.camera is None:
            return
        
        ret, frame = self.camera.read()
        if ret:
            # 转换颜色空间 BGR -> RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # 转换为PIL Image
            image = Image.fromarray(frame_rgb)
            
            # 计算缩放比例
            max_width = self.image_frame.winfo_width() - 40
            max_height = self.image_frame.winfo_height() - 40
            
            if max_width <= 0:
                max_width = 500
            if max_height <= 0:
                max_height = 500
            
            # 等比例缩放
            image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
            
            # 转换为PhotoImage
            photo = ImageTk.PhotoImage(image)
            
            # 显示图片
            self.image_label.config(image=photo, text='')
            self.image_label.image = photo
            
            # 如果启用了实时识别，定期进行OCR
            if hasattr(self, 'real_time_recognition') and self.real_time_recognition:
                current_time = time.time()
                if current_time - self.last_recognition_time >= self.recognition_interval:
                    self.last_recognition_time = current_time
                    # 保存当前帧并在主线程中识别
                    self.current_frame = frame_rgb.copy()
                    self.root.after(10, self.recognize_current_frame)
        
        # 继续更新（约30fps）
        if self.camera_running:
            self.root.after(33, self.update_camera_feed)
    
    def recognize_current_frame(self):
        """识别当前摄像头帧（后台线程，不阻塞UI）"""
        # 如果上一次识别还没完成，跳过
        if getattr(self, '_recognizing', False):
            return
        if not self.ocr_engine or not hasattr(self, 'current_frame'):
            return

        self._recognizing = True

        def do_recognize():
            try:
                import tempfile, os
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    tmp_path = tmp.name
                    Image.fromarray(self.current_frame).save(tmp_path, quality=85)

                texts = self.call_ocr_worker(tmp_path)
                try:
                    os.unlink(tmp_path)
                except:
                    pass

                if texts:
                    dates = self.extract_dates(texts)
                    if dates:
                        self.root.after(0, self.display_results, texts, dates)
            except:
                pass
            finally:
                self._recognizing = False

        threading.Thread(target=do_recognize, daemon=True).start()
    
    def recognize_camera_frame(self, frame):
        """识别摄像头帧 - 已废弃，改用主线程机制"""
        pass


def main():
    root = tk.Tk()
    app = ModernOCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
