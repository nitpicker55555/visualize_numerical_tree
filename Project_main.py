import pyvista as pv
import numpy as np
from pyvista import UnstructuredGrid,CellType
import copy
# 定义楔形的顶点
from matplotlib.colors import ListedColormap
#生成测试嵌套数据 retrun 嵌套列表
def random_generate():

    # 定义数列长度和变化速度参数
    N = 20
    a = 1

    # 生成等差数列
    x = np.linspace(0, 1, N)

    # 生成二次函数变化的数列
    y = -a * x ** 2 + a * x

    # 将数列归一化到[0, 1]范围内
    y = (y - np.min(y)) / (np.max(y) - np.min(y))

    # 将数列转换为先降低再变高的形式
    y = 10 - 6 * (y)
    result = []
    # 输出结果
    print(list(y))
    for i in list(y):
        result.append([])
        result[-1].append(i)
    print(result)

    bias = 0.4
    result_ = []
    # 1
    result_.append([])
    result_[-1].append(result[0][0])
    result_.append([])
    result_[-1].append(result[1][0])
    # 2 1
    result_.append([])
    result_[-1].append([result[2][0], result[2][0] + bias])

    for bbb in range(len(result)):

        if bbb >= 3:
            if (bbb-1) % 2 == 0:
                aaa = int((bbb-1) / 2)
            result_.append([])
            temp = result_
            for i in range(aaa):
                temp = temp[-1]
                temp.append([])
                temp[-1].append(result[bbb][0] + (i * bias))
                if i == aaa-1:
                    temp[-1].append([result[bbb][0] + (i * bias), result[bbb][0] + (i * bias * 2)])
    return (result_)
#建立单个三棱柱模型
def create_edge(z,x_width,x_coordinate,y,z_coordinate,cube_hight,y_positivity=True):

    #z=list数值，x_width 每次分支宽度减半，x_coordinate 按整体顺序，y 按节点顺序，z_coordinate：zi=xi+|z(i-1)|
    #z:直角边高度,x_width 直角边宽度，x_coordinate: x轴顺序，y:横轴顺序，z_coordinate:z轴顺序，zi=xi+|z(i-1)|
    if isinstance(cube_hight,list):
        cube_hight_result=cube_hight[0]
    else:
        cube_hight_result=cube_hight
    print(cube_hight_result, "cube_hight")
    points = np.array(
        [[0.0, 0.0, 0.0],
         [1.0, 0.0, 0.0],
         [1.0, 1.0, 0.0],
         [0.0, 1.0, 0.0],
         [0.0, 0.0, 1.0],
         [1.0, 0.0, 1.0],
         [1.0, 1.0, 1.0],
         [0.0, 1.0, 1.0]]
    )
    points[:, -1] *= cube_hight_result
    points[:, 0] *= x_width
    points[:, 0] += x_coordinate
    points[:,1]+=y
    points[:, 2] += z_coordinate-cube_hight_result

    cells = [len(points)] + list(range(len(points)))

    ex_cube = UnstructuredGrid(cells, [CellType.HEXAHEDRON], points)
    ex_cube['Hight_of_cube'] = ex_cube.points[:, -1]



    points = np.array(
        [[0., 0, z],
         [0., 1, 0.],
         [0., 0, 0.],
         [x_width, 0, z],
         [x_width, 1, 0.],
         [x_width, 0, 0.]]
    )
    if y_positivity!=True:
        points[:,1]*=-1
        points[:,1]+=1
    points[:,1]+=y
    points[:, -1] += z_coordinate
    points[:, 0] += x_coordinate
    print(list(np.arange(len(points))))
    cells = [len(points)] + list(range(len(points)))
    ex = UnstructuredGrid(cells, [CellType.WEDGE], points)
    plotter.add_point_labels(
        ex.cell_centers().points,
        [str(y)+"_"+str(x_coordinate)[0]],
        always_visible=True,
        font_size=10,
    )

    ex['Hight_of_cube'] = ex.points[:, -1]

    plotter.add_mesh(ex,
        #ex.explode(0.5),
                    specular=1,#反光

    smooth_shading = True,
                       split_sharp_edges = True,
    lighting=True,

    scalar_bar_args={'title': "Height_of_wedge"},

                     )
    plotter.add_mesh(ex_cube,
                     opacity=0.1,
                     #ex.explode(0.5),
                    # specular=1,#反光

    #smooth_shading = True,
    #                   split_sharp_edges = True,
    #lighting=True,



                     )
#查找宽度位置顺序，输入宽度，以及上个段落的宽度列表，retrun 母节点位置
def find_index(a, b):
    """
    在列表a中查找数字b的位置
    :param a: 一个数字列表
    :param b: 要查找的数字
    :return: 该数字在列表中的位置
    """
    sum_=[]
    for i in range(len(a)):
        sum_.append(sum(a[:i]))
        if b < sum_[i]:
            print(sum_)
            return i - 1
    print(sum_)
    return len(a) - 1
#输入三棱柱高度，retrun block_dict 包含段落顺序：[宽度，高度]
def create_dict(wedge_hight):

    def get_index_list(lst):
        index_list = []
        for i, val in enumerate(lst):
            if isinstance(val, list):  # 如果是列表就再传进来
                sub_index_list = get_index_list(val)
                for sub_index in sub_index_list:  # 遍历列表中的列表
                    index_list.append([i] + sub_index)  # 最终输出
            else:  # 如果不是列表则记下序号
                index_list.append([i])
        return index_list

    # [[0, 0], [1, 0, 0], [1, 0, 1], [2, 0, 0, 0], [2, 0, 0, 1], [2, 0, 1], [3, 0, 0, 0, 0], [3, 0, 0, 0, 1], [3, 0, 0, 1], [3, 0, 1]]
    # [1, 0, 0], [1, 0, 1]：整体序号为1，上一整体序号中节点序号0的变体，变体序号，
    # 如果深度增加则是变体
    # [2, 0, 1, 0], [2, 0, 1, 1]：中间0, 1就是 [1, 0, 1]中后两位的索引


    # z=list数值，x_width 每次分支宽度减半，x_coordinate 按整体顺序，y 按节点顺序，z_coordinate：zi=xi+|z(i-1)|

    depth = {}  # 每个段节点的数量
    index_list = get_index_list(wedge_hight)  # 每个节点位置
    # print(index_list)
    for node_index in index_list:
        if node_index[0] not in depth:
            depth[node_index[0]] = 0
        else:
            depth[node_index[0]] += 1
    # 分支
    # 顺序+1 and node[1:-2].append()
    # 分支顺序排列
    # 整体顺序+1

    sum_lists = []  # 两层嵌套，第一层段落号，第二层node_list
    block_dict = {}
    # 判断分支，输出分支

    for paragraph_num, nodes in enumerate(wedge_hight):  # 每一段遍历。i:段落号

        node_list = get_index_list(nodes)  # 每一段的非列表元素index


        sum_lists.append(node_list)
        print(sum_lists)
        print("每一段的非列表元素index:", node_list, "value:----", nodes)

        if paragraph_num != 0:
            split_node = []
            main_node_list = copy.copy(sum_lists[-2])  # 在上一个段落的node_list中查找母节点

            for section_num, node in enumerate(node_list):  # ii小节号
                if node not in sum_lists[-2]:  # 如果当前小节不在之前的段落之中
                    split_node.append(node)  # 当前小节属于分叉节点
                else:
                    pop_index = main_node_list.index(node)
                    print("main_node_list", main_node_list, "pop_index", pop_index, "node", node)
                    main_node_list.pop(pop_index)  # 如果当前小节在之前的段落之中，剔除之前的段落之中重复小节获得 分叉点的母节点
            if split_node!=[]:    #如果是空集那就没有分叉
                split_value = index2num(split_node, nodes)
                main_value = index2num(main_node_list, wedge_hight[paragraph_num - 1])
            else:
                split_value=[]
                main_value = None



            for section_num, node in enumerate(node_list):  # ii小节号
                if node in split_node:
                    # print("main_node_list",main_node_list,",split_node",split_node,",node",node,"round(split_node.index(node)/2)",round((split_node.index(node)+1)/2))
                    print("")
                    # width_ii 找到分支节点对应的母节点ii小节号，分支节点序号/2 对应 母节点序列序号
                    width_ii = sum_lists[-2].index(main_node_list[int((split_node.index(node)) / 2)])
                    print("width_ii", width_ii)
                    print("width", block_dict["{}_{}".format(paragraph_num - 1, width_ii)])
                    print("value_split", split_value[split_node.index(node)])
                    block_dict["{}_{}".format(paragraph_num, section_num)] = [
                        block_dict["{}_{}".format(paragraph_num - 1, width_ii)][0] / 2,
                        split_value[split_node.index(node)],index2num(node,cube_list[paragraph_num])]
                else:
                    # 宽度和上一个段落一致
                    print("node", node, "nodes", nodes)
                    width_ii = sum_lists[-2].index(node)
                    print("value_normal", index2num(node, nodes))
                    print("width_normal", [block_dict["{}_{}".format(paragraph_num - 1, width_ii)]])
                    block_dict["{}_{}".format(paragraph_num, section_num)] = [
                        block_dict["{}_{}".format(paragraph_num - 1, width_ii)][0], index2num(node, nodes),index2num(node,cube_list[paragraph_num])]

            print(main_node_list, "----", split_node)
            print(main_value, "----", split_value)


        else:
            x_width_first=5 #第一段初始x_width=2
            for section_num in range(len(nodes)):

                print(x_width_first / len(nodes), "width ")
                block_dict["{}_{}".format(paragraph_num, section_num)] = [x_width_first / len(nodes),
                                                                          nodes[section_num],cube_list[paragraph_num][section_num]]  # 记录第一个段落宽度，高度
    return (block_dict)
#输入字典，字典键值是非列表数值索引，value=非列表数值，return 按照索引和对应数值生成的列表
def generate_list(a):
    result = []
    for key, value in a.items():
        index = [int(i) for i in key.strip('[]').split(', ')]
        while len(result) <= index[0]:
            result.append([])
        if len(index) == 1:
            result[index[0]].append(value)
        else:
            temp_list = result[index[0]]
            for i in index[1:-1]:
                while len(temp_list) <= i:
                    temp_list.append([])
                temp_list = temp_list[i]
            while len(temp_list) <= index[-1]:
                temp_list.append(None)
            temp_list[index[-1]] = value
        if len(key)==3:
            result=result[0]
    return result
#输入嵌套列表，return 非列表数值索引
def get_index_list(lst):
    index_list = []
    for i, val in enumerate(lst):
        if isinstance(val, list):  # 如果是列表就再传进来
            sub_index_list = get_index_list(val)
            for sub_index in sub_index_list:  # 遍历列表中的列表
                index_list.append([i] + sub_index)  # 最终输出
        else:  # 如果不是列表则记下序号
            index_list.append([i])
    #print(index_list)
    return index_list
#给嵌套列表index，retrun 数值列表，被flatten的嵌套列表
def index2num(indexs, given_list):

    # ([[1]], [[[0.2, 0.3], 0.3], 0.1]) ---- ([[1, 0], [1, 1]], [[[0.2, 0.3], 0.3], [0.2, 0.3]])
    result = []
    if isinstance(indexs[0], list):
        a = "complex"  #嵌套列表索引列表，好几个索引
        for index in indexs:
            temp = given_list
            for i in index:
                temp = temp[i]
            result.append(temp)
    else:
        a = "simple" #嵌套列表索引
        temp = given_list
        for index in indexs:
            temp = temp[index]
        result.append(temp)
    print(indexs, "given_list", given_list, "result", result, a)

    return result
#输入两个相邻段落，（嵌套高度列表），输出三棱柱高度嵌套列表
def one_element(a,b):
    index_list_a=get_index_list(a) #输入嵌套列表，return 非列表数值索引
    index_list_b=get_index_list(b)
    flatten_num_a=index2num(index_list_a,a) #被flatten的嵌套列表
    flatten_num_b=index2num(index_list_b,b)
    #print(index2num(get_index_list(a),a))
    #print(index2num(get_index_list(b),b))
    result={}
    cube_hight={}
    for i in range(len(index_list_b)):
        if index_list_b[i] in index_list_a: #如果相同的数值索引在相邻段落再次出现，则该数值节点没有分叉
            for ii in range(len(index_list_a)):  #a是前一个段落
                if index_list_a[ii] == index_list_b[i]:   #找到没有分叉的节点
                    result[str(index_list_b[i])] = flatten_num_a[ii]-flatten_num_b[i]  #高度相减
                    cube_hight[str(index_list_b[i])] = min(flatten_num_a[ii],flatten_num_b[i])
        else:
            for ii in range(len(index_list_a)):
                if index_list_a[ii][1:]==index_list_b[i][1:-1]:  #找到分叉节点 前一个高度嵌套列表索引 第一个到倒数第一个数值和 后一个段落的 第一个到倒数第二个数值一样
                    print(index_list_a[ii],index_list_a[ii][1:])
                    print(index_list_b[i],index_list_b[i][1:-1])
                    result[str(index_list_b[i])]=flatten_num_a[ii]-flatten_num_b[i]    #高度相减
                    cube_hight[str(index_list_b[i])] = min(flatten_num_a[ii], flatten_num_b[i])
    list_result=generate_list(result)
    cube_list=generate_list(cube_hight)
    print(result,"____",list_result)
    return list_result,cube_list
#遍历高度列表，return 三棱柱高度列表
def process_hightlist(hight_list):
    result_list_final = []
    cube_list=[]
    for i in range(len(hight_list) - 1):
        one_element_result=one_element(hight_list[i], hight_list[i + 1])
        result_list_final.append(one_element_result[0])
        cube_list.append(one_element_result[1])
    print("result_list_final",result_list_final)
    return result_list_final,cube_list
# z=list数值，x_width 每次分支宽度减半
# z:直角边高度,x_width 直角边宽度，x_coordinate: x轴起始坐标,(sum x_width(i-1))，y:横轴顺序，z_coordinate:z轴顺序，z_coordinate i= z_coordinate (i-1)-zi，y_positivity:坡向上还是向下
#计算其他参数
def wedge_claculate():
    coordinate_dict = {}
    width = []
    paragraph_num = 0
    width_pre = []
    for paragraph in block_dict:
        if int(paragraph.split("_")[0]) > paragraph_num:  # 换到新的段落
            width_pre = copy.copy(width)
            width = []  # 新段落清空width
        paragraph_num = int(paragraph.split("_")[0])

        if isinstance(block_dict[paragraph][1], list):
            z_float = block_dict[paragraph][1][0]
        else:
            z_float = block_dict[paragraph][1]
        z = z_float

        x_width = block_dict[paragraph][0]
        x_coordinate = sum(width)  ##x轴起始坐标,(sum x_width(i-1))，
        y = int(paragraph.split("_")[0]) + 1

        width.append(block_dict[paragraph][0])  # x_width列表，用于求和，

        if int(paragraph.split("_")[0]) != 0:
            main_node_section = find_index(width_pre, x_coordinate)
            z_coordinate_pre = coordinate_dict["{}_{}".format(int(paragraph_num) - 1, main_node_section)][-2]
            y_positivity_pre = coordinate_dict["{}_{}".format(int(paragraph_num) - 1, main_node_section)][-1]
            z_pre = coordinate_dict["{}_{}".format(int(paragraph_num) - 1, main_node_section)][0]
            if z < 0:
                z = abs(z)
                y_positivity = False
                if y_positivity_pre == False:  # 上坡转上坡

                    z_coordinate = (z_coordinate_pre + z_pre)
                else:  # 下坡转上坡
                    z_coordinate = (z_coordinate_pre)

            else:
                y_positivity = True
                if y_positivity_pre == False:  # 上坡转下坡
                    z_coordinate = z_pre - z + z_coordinate_pre
                else:
                    z_coordinate = (z_coordinate_pre - z)

        else:
            z_coordinate = 0
            if z < 0:
                z = abs(z)
                y_positivity = False
            else:
                y_positivity = True

        print("---", z, x_width, x_coordinate, y, z_coordinate,block_dict[paragraph][2])
        create_edge(z, x_width, x_coordinate, y, z_coordinate, block_dict[paragraph][2],y_positivity)

        coordinate_dict[paragraph] = [z, x_width, x_coordinate, y, z_coordinate, y_positivity]
"""
plotter.enable_ssao(radius=15, bias=0.5)
plotter.enable_anti_aliasing('ssaa')
plotter.camera.zoom(1.7)
/
plotter.enable_ssao(kernel_size=128)
plotter.enable_anti_aliasing('ssaa')
"""
plotter = pv.Plotter()
#content_list = [[10.0],[12],[12],[12]]
hight_list = random_generate()#高度列表
process_hightlist_result=process_hightlist(hight_list)#三棱柱高度列表
wedge_hight=process_hightlist_result[0]
cube_list=process_hightlist_result[1]
print(cube_list,"asdasdasdasd")
block_dict=create_dict(wedge_hight)   #把高度换算为每个三棱柱的高度，

wedge_claculate()
plotter.show_axes_all()

##plotter.enable_ssao(kernel_size=128)
#plotter.enable_anti_aliasing('ssaa')

plotter.show()