import tkinter as tk
from tkinter import ttk
from tkinter import simpledialog, messagebox
import math
import time
from tkinter import filedialog
import json
from openai import OpenAI
from sorting_visualizer import SortingVisualizer # <--- 新增

# --- 模型部分 (Model) ---
# 负责图的内部数据结构和算法逻辑
class GraphModel:
    def __init__(self):
        self.nodes = {}  # 存储节点信息 { 'A': {'x': 50, 'y': 50}, 'B': ... }
        self.edges = []  # 存储边信息 [('A', 'B'), ('B', 'C')]
        self.adj_matrix = [] # 邻接矩阵
        self.node_map = []   # 节点名到矩阵索引的映射

    def add_node(self, node_name, x, y):
        if node_name in self.nodes:
            return False # 节点已存在
        self.nodes[node_name] = {'x': x, 'y': y}
        self._update_adj_matrix() # 添加节点后更新矩阵
        return True

    def add_edge(self, node1, node2, weight): # 新增 weight 参数
        if node1 not in self.nodes or node2 not in self.nodes:
            return False
        edge = tuple(sorted((node1, node2)))
    
        # 检查是否已存在（不考虑权重）
        existing_edges = [e[:2] for e in self.edges]
        if edge not in existing_edges:
            self.edges.append(edge + (weight,)) # 存入 ('A', 'B', 5) 这样的元组
            self._update_adj_matrix()
            return True
        return False

    def _update_adj_matrix(self):
        self.node_map = sorted(self.nodes.keys())
        size = len(self.node_map)
        # 矩阵中用0表示不通，实际权重表示连通
        self.adj_matrix = [[0] * size for _ in range(size)]
    
        map_to_idx = {name: i for i, name in enumerate(self.node_map)}

        # 注意这里的解包
        for n1, n2, weight in self.edges:
            if n1 in map_to_idx and n2 in map_to_idx:
                idx1 = map_to_idx[n1]
                idx2 = map_to_idx[n2]
                self.adj_matrix[idx1][idx2] = weight
                self.adj_matrix[idx2][idx1] = weight

    # 在 GraphModel 类中添加以下方法
    def get_adj_list(self):
        """根据当前的节点和边，生成邻接表"""
        adj_list = {node: [] for node in self.nodes}
        for n1, n2, _ in self.edges:
            adj_list[n1].append(n2)
            adj_list[n2].append(n1)
        return adj_list

    def dfs_traversal(self, start_node):
        """深度优先遍历的生成器，每一步都yield状态"""
        if start_node not in self.nodes:
            return
    
        adj_list = self.get_adj_list()
        stack = [start_node]
        visited = set()
    
        while stack:
            node = stack.pop()
            if node not in visited:
               visited.add(node)
               # yield状态：正在访问这个节点
               yield {'type': 'visit', 'node': node}
            
            # 把邻居逆序放入栈，以保证字典序小的先被访问
            for neighbor in sorted(adj_list[node], reverse=True):
                if neighbor not in visited:
                    # yield状态：正在探索这条边
                    yield {'type': 'explore', 'from': node, 'to': neighbor}
                    stack.append(neighbor)

    def bfs_traversal(self, start_node):
        """广度优先遍历的生成器，每一步都yield状态"""
        if start_node not in self.nodes:
            return

        adj_list = self.get_adj_list()
        queue = [start_node]
        visited = {start_node}

        # yield初始状态
        yield {'type': 'visit', 'node': start_node}

        while queue:
            node = queue.pop(0)
            for neighbor in sorted(adj_list[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    # yield状态：正在探索这条边
                    yield {'type': 'explore', 'from': node, 'to': neighbor}
                    # yield状态：正在访问这个节点
                    yield {'type': 'visit', 'node': neighbor}
                    queue.append(neighbor)

    # （3)在 GraphModel 类中添加以下两个方法
    def delete_node(self, node_name):
        """删除一个节点以及所有与之相关的边"""
        if node_name not in self.nodes:
            return False

        # 1. 从节点字典中删除节点
        del self.nodes[node_name]

        # 2. 从边列表中删除所有与该节点相关的边
        # 创建一个新的边列表，只包含与被删除节点无关的边
        self.edges = [edge for edge in self.edges if node_name not in edge]
    
        # 3. 更新邻接矩阵
        self._update_adj_matrix()
        return True

    def delete_edge(self, node1, node2):
        """删除一条边"""
        if node1 not in self.nodes or node2 not in self.nodes:
            return False
        
        edge_to_remove = tuple(sorted((node1, node2)))
        if edge_to_remove in self.edges:
            self.edges.remove(edge_to_remove)
            self._update_adj_matrix()
            return True
        return False
    
    # 在 GraphModel 类中添加 prim_mst 方法
    def prim_mst(self, start_node):
        """Prim算法生成最小生成树，并yield每一步"""
        if start_node not in self.nodes or not self.edges:
            return
        
        visited = {start_node}
        mst_edges = []
        # 只要已访问的节点数小于总节点数
        while len(visited) < len(self.nodes):
            possible_edges = []
            # 找到所有连接已访问节点和未访问节点的边
            for n1, n2, weight in self.edges:
                if (n1 in visited and n2 not in visited) or \
                    (n2 in visited and n1 not in visited):
                    possible_edges.append((n1, n2, weight))
        
            # 如果没有更多可连接的边（图不连通），则结束
            if not possible_edges:
                break
            
            # 找到这些边里权重最小的一条
            min_edge = min(possible_edges, key=lambda edge: edge[2])
        
            # yield 状态：我们选中了这条边
            yield {'type': 'mst_edge', 'edge': min_edge}
        
            mst_edges.append(min_edge)
        
            # 将新节点加入已访问集合
            if min_edge[0] not in visited:
                visited.add(min_edge[0])
            if min_edge[1] not in visited:
                visited.add(min_edge[1])
            
    def dijkstra_shortest_path(self, start_node, end_node):
        if start_node not in self.nodes or end_node not in self.nodes:
            return

        # 初始化距离字典，起点为0，其余为无穷大
        distances = {node: float('inf') for node in self.nodes}
        distances[start_node] = 0
        # 记录路径的前一个节点
        previous_nodes = {node: None for node in self.nodes}
        # 未访问的节点集合
        unvisited = set(self.nodes.keys())

        while unvisited:
            # 从未访问的节点中找到距离最小的节点
            current_node = min(unvisited, key=lambda node: distances[node])
        
            # 如果当前最小距离是无穷大，说明剩下的点不可达，结束
            if distances[current_node] == float('inf'):
                break

            unvisited.remove(current_node)
        
            # yield 状态：正在访问当前节点
            yield {'type': 'visit_dijkstra', 'node': current_node}

            # 如果到达终点，可以提前结束
            if current_node == end_node:
                break

            # 更新邻居的距离
            for n1, n2, weight in self.edges:
                neighbor = None
                if n1 == current_node and n2 in unvisited:
                    neighbor = n2
                elif n2 == current_node and n1 in unvisited:
                    neighbor = n1
            
                if neighbor:
                    new_dist = distances[current_node] + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        previous_nodes[neighbor] = current_node
                        # yield 状态：找到了更短的路径
                        yield {'type': 'relax', 'from': current_node, 'to': neighbor, 'dist': new_dist}

        # 算法结束后，回溯路径
        path = []
        current = end_node
        while current is not None:
            path.insert(0, current)
            current = previous_nodes[current]
    
        if path and path[0] == start_node:
            # yield 最终路径
            yield {'type': 'final_path', 'path': path, 'dist': distances[end_node]}
        else:
            yield {'type': 'no_path'}


    




# --- 视图与控制器部分 (View & Controller) ---
# 请用这个完整的、整理好的版本替换你现有的 __init__ 方法

# graph_visualizer.py

# 请用这个完整的、整理好的版本替换你现有的 __init__ 方法

class GraphVisualizerFrame(tk.Frame):
    def __init__(self, parent, menubar):
        super().__init__(parent) # 这是唯一需要的super()调用

        # --- 步骤1: 设置从主窗口传来的菜单栏 ---
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="文件", menu=file_menu)
        file_menu.add_command(label="保存", command=self._save_file)
        file_menu.add_command(label="加载", command=self._load_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self.winfo_toplevel().quit)

        # --- 步骤2: 初始化数据模型 ---
        self.model = GraphModel()
        self.node_radius = 20

        # --- 步骤3: 创建所有界面控件 (注意，这里只创建，不打包) ---

        # 3.1 左侧控制面板
        control_frame = tk.Frame(self)
        tk.Label(control_frame, text="操作面板").pack()
        tk.Button(control_frame, text="添加顶点", command=self._add_node_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="添加边", command=self._add_edge_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="删除顶点", command=self._delete_node_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="删除边", command=self._delete_edge_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="最小生成树(Prim)", command=self._mst_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="最短路径(Dijkstra)", command=self._dijkstra_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="深度优先遍历(DFS)", command=self._dfs_handler).pack(fill=tk.X, pady=5)
        tk.Button(control_frame, text="广度优先遍历(BFS)", command=self._bfs_handler).pack(fill=tk.X, pady=5)

        # 3.2 右侧信息显示区
        info_frame = tk.Frame(self)
        tk.Label(info_frame, text="数据结构表示").pack()
        self.info_text = tk.Text(info_frame, width=30, height=20)
        self.info_text.pack()

        # 3.3 底部 DSL 和 LLM 输入区
        input_frame = tk.Frame(self)
        self.dsl_text = tk.Text(input_frame, height=5)
        dsl_button_frame = tk.Frame(input_frame)
        tk.Button(dsl_button_frame, text="从代码绘制", command=self._draw_from_dsl).pack(fill=tk.X)
        tk.Button(dsl_button_frame, text="自然语言绘制(LLM)", command=self._draw_from_llm).pack(fill=tk.X)
        
        # 3.4 中间的画布 (它是最后一个，用来填充剩余空间)
        self.canvas = tk.Canvas(self, bg="white")


        # --- 步骤4: 按正确的顺序打包所有控件！ ---
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        # 打包底部的内部控件
        self.dsl_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        dsl_button_frame.pack(side=tk.RIGHT, padx=5)

        # 最后，让画布填充所有剩下的空间
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=(0, 10))


    def _add_node_handler(self):
        """控制器：处理添加顶点按钮点击事件"""
        node_name = simpledialog.askstring("输入", "请输入顶点名称 (e.g., A):")
        if not node_name:
            return
        
        # 自动计算新节点的位置（这里简单地围成一圈）
        count = len(self.model.nodes)
        center_x, center_y = 300, 250
        radius = 150
        angle = (2 * math.pi / max(1, count + 1)) * count
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)

        if self.model.add_node(node_name, x, y):
            self.redraw_all()
        else:
            messagebox.showerror("错误", f"顶点 '{node_name}' 已存在!")

    def _add_edge_handler(self):
        node1 = simpledialog.askstring("输入", "请输入第一个顶点名称:")
        if not node1: return
        node2 = simpledialog.askstring("输入", "请输入第二个顶点名称:")
        if not node2: return
        # 新增：让用户输入权重
        weight = simpledialog.askinteger("输入", "请输入边的权重 (整数):", minvalue=1)
        if weight is None: return

        if self.model.add_edge(node1, node2, weight):
            self.redraw_all()
        else:
            messagebox.showerror("错误", "无法添加边，请检查顶点是否存在或边是否已存在。")

    def redraw_all(self):
        """视图：根据模型数据重绘所有内容"""
        self.canvas.delete("all")
        
        # 1. 画边
        for node1, node2, weight in self.model.edges: # 解包时加上 weight
            x1, y1 = self.model.nodes[node1]['x'], self.model.nodes[node1]['y']
            x2, y2 = self.model.nodes[node2]['x'], self.model.nodes[node2]['y']
            self.canvas.create_line(x1, y1, x2, y2, fill="gray", width=2)
            # 在边的中点绘制权重
            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
            self.canvas.create_text(mid_x, mid_y - 10, text=str(weight), fill="darkblue", font=("Arial", 10))
            
        # 2. 画顶点
        for name, data in self.model.nodes.items():
            x, y = data['x'], data['y']
            self.canvas.create_oval(x - self.node_radius, y - self.node_radius,
                                    x + self.node_radius, y + self.node_radius,
                                    fill="lightblue", outline="black")
            self.canvas.create_text(x, y, text=name, font=("Arial", 12, "bold"))

        # 3. 更新信息显示区的邻接矩阵
        self.update_info_display()

    def update_info_display(self):
        """视图：更新右侧的文本框来显示邻接矩阵和邻接表"""
        self.info_text.delete("1.0", tk.END)
        
        # --- 可视化邻接矩阵 ---
        self.info_text.insert(tk.END, "--- 邻接矩阵 ---\n")
        if not self.model.node_map:
            self.info_text.insert(tk.END, "(空)\n")
        else:
            header = "    " + "  ".join(self.model.node_map) + "\n"
            self.info_text.insert(tk.END, header)
            for i, row in enumerate(self.model.adj_matrix):
                row_str = f"{self.model.node_map[i]} | " + "  ".join(map(str, row)) + "\n"
                self.info_text.insert(tk.END, row_str)

        # --- 可视化邻接表 (留作下一步) ---
        self.info_text.insert(tk.END, "\n\n--- 邻接表 ---\n")
        adj_list = self.model.get_adj_list()
        if not adj_list:
            self.info_text.insert(tk.END, "(空)\n")
        else:
            for node, neighbors in adj_list.items():
                line = f"{node} -> {' -> '.join(neighbors)}\n"
                self.info_text.insert(tk.END, line)


    # 在 GraphVisualizer 类中，添加以下方法
    def _dfs_handler(self):
        start_node = simpledialog.askstring("输入", "请输入遍历起始顶点:")
        if start_node and start_node in self.model.nodes:
            traversal_steps = self.model.dfs_traversal(start_node)
            self._play_animation(traversal_steps)
        else:
            messagebox.showerror("错误", "起始顶点不存在！")
    
    def _bfs_handler(self):
        start_node = simpledialog.askstring("输入", "请输入遍历起始顶点:")
        if start_node and start_node in self.model.nodes:
            traversal_steps = self.model.bfs_traversal(start_node)
            self._play_animation(traversal_steps)
        else:
            messagebox.showerror("错误", "起始顶点不存在！")

    def _play_animation(self, steps):
        """核心动画播放器"""
        # 先重置所有颜色
        self.redraw_all()
        
        # 禁用按钮，防止动画期间误操作
        for child in self.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Button):
                        btn.config(state=tk.DISABLED)
                    
        # 逐帧播放
        for step in steps:
            if step['type'] == 'visit':
                node_name = step['node']
                data = self.model.nodes[node_name]
                x, y = data['x'], data['y']
                # 高亮访问过的顶点
                self.canvas.create_oval(x - self.node_radius, y - self.node_radius,
                                        x + self.node_radius, y + self.node_radius,
                                        fill="lightgreen", outline="black")
                self.canvas.create_text(x, y, text=node_name, font=("Arial", 12, "bold"))
        
            elif step['type'] == 'explore':
                n1_name, n2_name = step['from'], step['to']
                x1, y1 = self.model.nodes[n1_name]['x'], self.model.nodes[n1_name]['y']
                x2, y2 = self.model.nodes[n2_name]['x'], self.model.nodes[n2_name]['y']
                # 高亮正在探索的边
                self.canvas.create_line(x1, y1, x2, y2, fill="red", width=3, arrow=tk.LAST)

            elif step['type'] == 'mst_edge': # 这是新增的部分
                n1_name, n2_name, weight = step['edge']
                x1, y1 = self.model.nodes[n1_name]['x'], self.model.nodes[n1_name]['y']
                x2, y2 = self.model.nodes[n2_name]['x'], self.model.nodes[n2_name]['y']
                # 用醒目的颜色（如金色或深蓝色）高亮MST的边
                self.canvas.create_line(x1, y1, x2, y2, fill="gold", width=4)
                # 重新绘制两个端点，确保它们在最上层
                for name in [n1_name, n2_name]:
                    data = self.model.nodes[name]
                    x, y = data['x'], data['y']
                    self.canvas.create_oval(x - self.node_radius, y - self.node_radius,
                                            x + self.node_radius, y + self.node_radius,
                                            fill="lightgreen", outline="black")
                    self.canvas.create_text(x, y, text=name, font=("Arial", 12, "bold"))

            # 新增 Dijkstra 的可视化逻辑
            elif step['type'] == 'visit_dijkstra':
                # 用一种新颜色（比如紫色）标记被Dijkstra算法确定最短路径的节点
                node_name = step['node']
                data = self.model.nodes[node_name]
                x, y = data['x'], data['y']
                self.canvas.create_oval(x - self.node_radius, y - self.node_radius,
                                        x + self.node_radius, y + self.node_radius,
                                        fill="violet", outline="black")
                self.canvas.create_text(x, y, text=node_name)
        
            elif step['type'] == 'relax':
                # 用虚线或不同颜色表示正在“松弛”的边
                n1, n2 = step['from'], step['to']
                x1, y1 = self.model.nodes[n1]['x'], self.model.nodes[n1]['y']
                x2, y2 = self.model.nodes[n2]['x'], self.model.nodes[n2]['y']
                self.canvas.create_line(x1, y1, x2, y2, fill="orange", width=2, dash=(4, 4))
        
            elif step['type'] == 'final_path':
                # 用最醒目的颜色（如亮蓝色）高亮最终的最短路径
                path = step['path']
                total_dist = step['dist']
                for i in range(len(path) - 1):
                    n1, n2 = path[i], path[i+1]
                    x1, y1 = self.model.nodes[n1]['x'], self.model.nodes[n1]['y']
                    x2, y2 = self.model.nodes[n2]['x'], self.model.nodes[n2]['y']
                    self.canvas.create_line(x1, y1, x2, y2, fill="deepskyblue", width=5)
                messagebox.showinfo("完成", f"找到最短路径！\n路径: {' -> '.join(path)}\n总权重: {total_dist}")
                break # 找到路径就结束动画
            
            elif step['type'] == 'no_path':
                messagebox.showwarning("提示", "从起点无法到达终点！")
                break

            self.update() # 刷新屏幕
            time.sleep(0.8) # 暂停0.8秒，形成动画效果

        # 动画结束后恢复按钮
        for child in self.winfo_children():
            if isinstance(child, tk.Frame):
                for btn in child.winfo_children():
                    if isinstance(btn, tk.Button):
                        btn.config(state=tk.NORMAL)

        messagebox.showinfo("完成", "遍历已完成！")

    def _delete_node_handler(self):
        """控制器：处理删除顶点按钮点击事件"""
        node_name = simpledialog.askstring("输入", "请输入要删除的顶点名称:")
        if not node_name: return
        
        if self.model.delete_node(node_name):
            self.redraw_all() # 删除成功，重绘整个画布
        else:
            messagebox.showerror("错误", f"顶点 '{node_name}' 不存在!")

    def _delete_edge_handler(self):
        """控制器：处理删除边按钮点击事件"""
        node1 = simpledialog.askstring("输入", "请输入边的第一个顶点:")
        if not node1: return
        node2 = simpledialog.askstring("输入", "请输入边的第二个顶点:")
        if not node2: return

        if self.model.delete_edge(node1, node2):
            self.redraw_all() # 删除成功，重绘整个画布
        else:
            messagebox.showerror("错误", "边不存在或顶点输入有误!")


    def _mst_handler(self):
        start_node = simpledialog.askstring("输入", "请输入MST的起始顶点:")
        if start_node and start_node in self.model.nodes:
            mst_steps = self.model.prim_mst(start_node)
            self._play_animation(mst_steps)
        else:
            messagebox.showerror("错误", "起始顶点不存在！")


    def _save_file(self):
        """控制器：保存当前图的状态到文件"""
        # 弹出文件保存对话框，让用户选择保存位置和文件名
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return # 如果用户取消，则什么都不做

        # 准备要保存的数据：节点信息和边信息
        data_to_save = {
            "nodes": self.model.nodes,
            "edges": self.model.edges
        }

        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data_to_save, f, indent=4) # 使用json格式写入文件
            messagebox.showinfo("成功", f"文件已保存到:\n{filepath}")
        except Exception as e:
            messagebox.showerror("错误", f"保存文件失败: {e}")

    def _load_file(self):
        """控制器：从文件加载图的状态"""
        # 弹出文件打开对话框
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                loaded_data = json.load(f)

            # 将加载的数据更新到模型中
            self.model.nodes = loaded_data.get("nodes", {})
            self.model.edges = loaded_data.get("edges", [])
            
            # 重要：更新邻接矩阵并重绘
            self.model._update_adj_matrix()
            self.redraw_all()
            
            messagebox.showinfo("成功", f"文件已从:\n{filepath}\n加载成功!")
        except Exception as e:
            messagebox.showerror("错误", f"加载文件失败: {e}")

    def _dijkstra_handler(self):
        start_node = simpledialog.askstring("输入", "请输入最短路径的【起始】顶点:")
        if not start_node or start_node not in self.model.nodes:
            messagebox.showerror("错误", "起始顶点不存在！")
            return
        end_node = simpledialog.askstring("输入", "请输入最短路径的【终点】顶点:")
        if not end_node or end_node not in self.model.nodes:
            messagebox.showerror("错误", "终点顶点不存在！")
            return
        
        path_steps = self.model.dijkstra_shortest_path(start_node, end_node)
        self._play_animation(path_steps)

    # 在 GraphVisualizer 类中，使用这个最终、最强的版本
    def _draw_from_dsl(self):
        """
        解析DSL并绘制图 (最终版)
        - 支持单行或多行输入
        - 修复了节点重叠的bug
        """
        code = self.dsl_text.get("1.0", tk.END).strip()
    
        # 先清空现有模型
        self.model.nodes.clear()
        self.model.edges.clear()

        # --- 核心逻辑改变：不再关心换行，直接处理指令流 ---
        parts = code.split()
    
        # 1. 预扫描，统计总共有多少个 'node' 命令
        num_nodes_to_create = parts.count('node')
     
        node_count = 0
        try:
            i = 0
            while i < len(parts):
                command = parts[i].lower()
            
                if command == 'node' and i + 1 < len(parts):
                    node_name = parts[i+1]
                    if node_name not in self.model.nodes:
                        # 2. 使用正确的节点总数来计算角度
                        center_x, center_y, radius = 300, 250, 150
                        angle = (2 * math.pi / max(1, num_nodes_to_create)) * node_count
                        x = center_x + radius * math.cos(angle)
                        y = center_y + radius * math.sin(angle)
                        self.model.add_node(node_name, x, y)
                        node_count += 1
                    i += 2 # 跳过 'node' 和 <节点名>
            
                elif command == 'edge' and i + 3 < len(parts):
                    n1, n2, weight = parts[i+1], parts[i+2], int(parts[i+3])
                    self.model.add_edge(n1, n2, weight)
                    i += 4 # 跳过 'edge' 和三个参数
            
                else:
                    # 如果遇到无法识别的命令或参数不足，就跳过它，防止程序卡死
                    i += 1
        
            # 3. 最后统一重绘
            self.redraw_all()
        except Exception as e:
            messagebox.showerror("DSL解析错误", f"解析失败: {e}")

    # 在 GraphVisualizer 类中添加方法
    def _draw_from_llm(self):
        """使用LLM将自然语言转换为DSL并绘制 (适配 openai >= 1.0.0)"""
    
        # 1. 初始化客户端 (新版库的核心变化)
        try:
            # 你不再需要 openai.api_key = "..." 这种全局设置了
            # 直接在这里创建客户端，并传入你的key
            client = OpenAI(api_key="") # <--- 在这里填入你的API Key
        except NameError:
            messagebox.showerror("错误", "'OpenAI' 未定义。请确保文件顶部有 'from openai import OpenAI'。")
            return
        except Exception as e:
            messagebox.showerror("错误", f"创建OpenAI客户端失败: {e}")
            return

        user_prompt = self.dsl_text.get("1.0", tk.END).strip()
        if not user_prompt:
            messagebox.showinfo("提示", "请输入自然语言描述。")
            return

        system_message = (
            "你是一个图结构DSL生成器。将用户的自然语言描述转换为一个简单的DSL。"
            "DSL格式为：每行一个命令。'node <节点名>' 用于创建节点，'edge <节点1> <节点2> <权重>' 用于创建带权边。"
            "只输出DSL代码，不要有任何其他解释或注释。"
        )

        messagebox.showinfo("LLM请求", "正在向AI发送请求，请稍候...")
        self.update()

        try:
            # 2. 调用API的方式改变了
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ]
            )
            # 3. 获取返回内容的方式也变了
            dsl_code = response.choices[0].message.content.strip()
        
            self.dsl_text.delete("1.0", tk.END)
            self.dsl_text.insert("1.0", dsl_code)
            self._draw_from_dsl()

        except Exception as e:
            messagebox.showerror("LLM API错误", f"请求失败: {e}")


class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("数据结构与算法可视化模拟器")
        self.geometry("1200x700")

        # --- 创建主菜单栏 ---
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        # --- 创建选项卡控件 ---
        notebook = ttk.Notebook(self)
        notebook.pack(pady=10, padx=10, fill="both", expand=True)

        # --- 创建并添加第一个选项卡：图形结构 ---
        # 我们把主菜单栏 menubar 传递给它
        graph_frame = GraphVisualizerFrame(notebook, menubar)
        notebook.add(graph_frame, text="图形结构")

        # --- 创建并添加第二个选项卡：排序算法 ---
        sort_frame = SortingVisualizer(notebook)
        notebook.add(sort_frame, text="排序算法")


if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()