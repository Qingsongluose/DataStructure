# sorting_visualizer.py

import tkinter as tk
from tkinter import ttk
import random
import time # 确保文件顶部有 import time
from tkinter import messagebox # 确保文件顶部有这个

class SortingVisualizer(tk.Frame):
    """排序算法可视化的UI框架"""
    def __init__(self, parent):
        super().__init__(parent)

        # --- 数据模型 ---
        self.data = []
        self.bar_width = 15
        self.bar_gap = 5

        # --- UI 布局 ---
        control_frame = tk.Frame(self)
        control_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        self.canvas = tk.Canvas(self, bg='white')
        self.canvas.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        # --- 控制按钮 ---
        tk.Button(control_frame, text="生成新数组", command=self._generate_data).pack(side=tk.LEFT, padx=5)
        tk.Label(control_frame, text="选择排序算法:").pack(side=tk.LEFT, padx=5)
        
        # --- 排序算法按钮 (暂时是占位符) ---
        tk.Button(control_frame, text="直接插入排序", command=self._insertion_sort).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="简单选择排序", command=self._selection_sort).pack(side=tk.LEFT, padx=5)
        tk.Button(control_frame, text="快速排序", command=self._quick_sort).pack(side=tk.LEFT, padx=5)

        # --- 初始状态 ---
        self._generate_data()

    # 在 sorting_visualizer.py 文件中，用这个新版本替换旧的 _generate_data 方法

    def _generate_data(self):
        """生成一个随机的整数数组"""
    
        # 我们根据画布的实际宽度来决定数组的大小
        # 这里我们先假设一个初始宽度，稍后在绘图时会再次精确计算
        canvas_width = 800 
        canvas_height = 400 

        # --- 关键的修复在这里 ---
        # 必须使用 // (整型除法) 来确保 self.data_size 是一个整数！
        self.data_size = canvas_width // (self.bar_width + self.bar_gap)
    
        # 现在 self.data_size 是一个整数，range() 函数可以正常工作了
        self.data = [random.randint(10, canvas_height) for _ in range(self.data_size)]
     
        # 调用绘图函数来显示新生成的数组
        self._draw_data()

    def _draw_data(self, color_map=None):
        """根据当前数据绘制柱状图"""
        self.canvas.delete("all")
        if color_map is None:
            color_map = {}

        canvas_height = self.canvas.winfo_height()
        canvas_width = self.canvas.winfo_width()
        
        # 计算每个柱子的宽度和间隙
        bar_total_width = canvas_width / len(self.data)
        self.bar_width = bar_total_width * 0.8
        self.bar_gap = bar_total_width * 0.2
        
        for i, value in enumerate(self.data):
            x0 = i * (self.bar_width + self.bar_gap)
            y0 = canvas_height
            x1 = x0 + self.bar_width
            y1 = canvas_height - value
            
            color = color_map.get(i, "skyblue") # 默认颜色为天蓝色
            self.canvas.create_rectangle(x0, y0, x1, y1, fill=color, outline="black")
            self.canvas.create_text(x0 + self.bar_width / 2, y1 - 10, text=str(value), font=("Arial", 8))
            
    # --- 第1步：实现算法逻辑 (这是一个生成器) ---
    def _insertion_sort_logic(self):
        """直接插入排序的逻辑生成器"""
        data = self.data.copy()
        n = len(data)
    
        # 从第二个元素开始
        for i in range(1, n):
            key = data[i] # 当前要插入的元素
            j = i - 1
        
            # 产生一个高亮：绿色表示已排序部分，黄色表示当前要插入的牌
            color = {k: 'lightgreen' for k in range(i)}
            color[i] = 'gold'
            yield {'data': data.copy(), 'highlight': color}
            time.sleep(0.2)

            # 将已排序部分中大于 key 的元素向后移动
            while j >= 0 and data[j] > key:
                # 高亮比较的两个元素
                color[j] = 'red'
                yield {'data': data.copy(), 'highlight': color}
                time.sleep(0.3)
            
                data[j + 1] = data[j]
                color.pop(j) # 恢复颜色
                j -= 1
            
                # 显示移动后的状态
                yield {'data': data.copy(), 'highlight': color}
                time.sleep(0.3)

            data[j + 1] = key
        
            # 插入完成后的状态
            color = {k: 'lightgreen' for k in range(i + 1)}
            yield {'data': data.copy(), 'highlight': color}
            time.sleep(0.2)
        
        # 全部完成，所有都变绿色
        yield {'data': data.copy(), 'highlight': {k: 'lightgreen' for k in range(n)}}


    # --- 第2步：实现通用的动画播放器 ---
    def _play_sort_animation(self, steps_generator):
        """播放排序动画"""
        try:
            for step in steps_generator:
                self.data = step['data']
                self._draw_data(color_map=step['highlight'])
                self.update_idletasks() # 刷新UI
                time.sleep(0.1) # 短暂延时，形成动画效果
        
            messagebox.showinfo("完成", "排序已完成！")
        except Exception as e:
            messagebox.showerror("错误", f"动画播放时发生错误: {e}")

    # --- 第3步：修改按钮的命令 ---
    def _insertion_sort(self):
        """处理直接插入排序按钮的点击事件"""
        print("开始执行直接插入排序...")
        steps = self._insertion_sort_logic()
        self._play_sort_animation(steps)
        
    # --- 1. 添加选择排序的逻辑生成器 ---
    def _selection_sort_logic(self):
        """简单选择排序的逻辑生成器"""
        data = self.data.copy()
        n = len(data)

        for i in range(n):
            # 假设当前位置i的元素是最小的
            min_idx = i
        
            # 在i之后的部分寻找真正的最小值
            for j in range(i + 1, n):
                # 高亮：绿色是已排序区，金色是当前坑位，红色是比较的两个元素
                color = {k: 'lightgreen' for k in range(i)}
                color[min_idx] = 'blue' # 当前找到的最小值
                color[i] = 'gold' # 将要被填充的坑位
                color[j] = 'red' # 正在扫描的元素
                yield {'data': data.copy(), 'highlight': color}
                time.sleep(0.05)

                if data[j] < data[min_idx]:
                    min_idx = j # 更新最小值索引

            # 高亮将要交换的两个元素
            color = {k: 'lightgreen' for k in range(i)}
            color[i] = 'orange'
            color[min_idx] = 'orange'
            yield {'data': data.copy(), 'highlight': color}
            time.sleep(0.5)

            # 将找到的最小元素与位置i的元素交换
            data[i], data[min_idx] = data[min_idx], data[i]

            # 显示交换后的结果，位置i已排序
            color = {k: 'lightgreen' for k in range(i + 1)}
            yield {'data': data.copy(), 'highlight': color}
            time.sleep(0.2)
    
        # 全部完成，所有都变绿色
        yield {'data': data.copy(), 'highlight': {k: 'lightgreen' for k in range(n)}}


    # --- 2. 替换旧的占位符方法 ---
    def _selection_sort(self):
        """处理简单选择排序按钮的点击事件"""
        print("开始执行简单选择排序...")
        steps = self._selection_sort_logic()
        self._play_sort_animation(steps)

    # # --- 1. 添加快速排序的逻辑生成器 ---
    # 在 sorting_visualizer.py 文件中，用这个修正后的方法替换旧的 _quick_sort_logic

    def _quick_sort_logic(self):
        """快速排序的逻辑生成器 (迭代版)"""
        data = self.data.copy()
        n = len(data)

        # 用一个栈来模拟递归
        stack = [(0, n - 1)]
    
        sorted_indices = set()

        while stack:
            low, high = stack.pop()
        
            if low >= high:
                # 如果子数组有效,将其标记为已排序
                if 0 <= low < n: sorted_indices.add(low)
                if 0 <= high < n: sorted_indices.add(high)
                continue

            # --- 分区 (Partition) 操作 ---
            pivot = data[high] # 选择最后一个元素作为基准
            i = low - 1 # i 是小于基准的最后一个元素的索引

            # 高亮:紫色是基准,灰色是当前处理的子数组
            color = {k: 'gray' for k in range(low, high)}
            color[high] = 'purple' # Pivot
        
            # --- 关键修复在这里！加上 for 关键字 ---
            for idx in sorted_indices: 
                color[idx] = 'lightgreen' # 已排序部分
            
            yield {'data': data.copy(), 'highlight': color}
            time.sleep(0.3)

            for j in range(low, high):
                # 高亮扫描指针j和分界线i
                color[j] = 'red'
                if i >= low: color[i] = 'blue'
                yield {'data': data.copy(), 'highlight': color}
                time.sleep(0.1)

                if data[j] <= pivot:
                    i += 1
                    data[i], data[j] = data[j], data[i]
                    # 显示交换结果
                    yield {'data': data.copy(), 'highlight': color}
                    time.sleep(0.2)
            
                # 恢复颜色
                if i >= low: color[i] = 'blue'
                color[j] = 'gray'

            # 将基准元素放到正确的位置
            data[i + 1], data[high] = data[high], data[i + 1]
            pivot_final_index = i + 1
            sorted_indices.add(pivot_final_index) # 这个位置已经排好

            # 显示基准归位
            color = {k: 'gray' for k in range(low, high + 1)}
            for idx in sorted_indices: 
                color[idx] = 'lightgreen'
            yield {'data': data.copy(), 'highlight': color}
            time.sleep(0.5)

            # 将左右两个子数组的范围压入栈
            stack.append((low, pivot_final_index - 1))
            stack.append((pivot_final_index + 1, high))

        # 全部完成
        yield {'data': data.copy(), 'highlight': {k: 'lightgreen' for k in range(n)}}


    # --- 2. 替换旧的占位符方法 ---
    def _quick_sort(self):
        """处理快速排序按钮的点击事件"""
        print("开始执行快速排序...")
        steps = self._quick_sort_logic()
        self._play_sort_animation(steps)


