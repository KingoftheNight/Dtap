import os
import sys
import numpy as np
import math
now_path = os.getcwd()
file_path = os.path.dirname(__file__)
sys.path.append(file_path)
import time

# 创建定义列表
def visual_create_n_matrix(x=20, fill=0):
    out = []
    for i in range(x):
        out.append(fill)
    return out

# 创建元素列表
def visual_create_aa(tp='p'):
    if tp == 'n':
        return ['A', 'T', 'C', 'G']
    elif tp == 'd':
        return ['N', 'P', 'D', 'H', 'C']
    else:
        return ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


# 配置时间戳
def visual_timestamp():
    a = time.mktime(time.localtime())
    return str(int(a))

# 读取fasta文件
def weblogo_read(file):
    with open(file, 'r') as f:
        data = f.readlines()
    mid = ''
    for i in data:
        i = i.strip('\n')
        if '>' in i:
            mid += '\n' + i + '\n'
        else:
            mid += i
    out = []
    for i in mid[1:].split('\n'):
        if '>' not in i:
            out.append(i.strip('\n'))
    return out

# 数据转信息熵
def weblogo_check(value):
    H = 0
    for i in value:
        if i != 0:
            H += i*math.log(i, 2)
    Rseq = math.log(len(value), 2) + H
    out = []
    for i in value:
        if i != 0:
            out.append(Rseq*i)
        else:
            out.append(0)
    return out

# 数据信息熵提取
def weblogo_count(data, tp='p'):
    aa_index = visual_create_aa(tp=tp)
    # 频数
    count = []
    for i in range(len(data[0])):
        mid = visual_create_n_matrix(x=len(aa_index))
        for line in data:
            if line[i] in aa_index:
                mid[aa_index.index(line[i])] += 1
        count.append(mid)
    count = np.array(count)
    # 频率
    percent = []
    for i in range(len(count)):
        mid = []
        for j in range(len(count[i])):
            mid.append(count[i, j]/np.sum(count[i]))
        # 转信息熵
        mid = weblogo_check(mid)
        percent.append(mid)
    return percent

# 字体定义
def weblogo_path(tp='p'):
    # 碱基字体
    if tp == 'n':
        path = {
            'A':'M60,100H48l-7-30.6H19L12,100H0L25,0h10L60,100z M38.5,58.4l-8-35.3h-1l-8,35.3H38.5z',
            'T':'M60,11.1H36.7V100H23.3V11.1H0V0h60V11.1z',
            'C':'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z',
            'G':'M60,100h-9.7l-1.7-8c-1.5,2.3-3.9,4.2-7.1,5.7c-3.2,1.5-6.8,2.3-10.6,2.3c-8,0-15.1-3.7-21.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,2.9-31.2,8.6-40C14.3,4.4,22.1,0,32,0c8.8,0,15.6,3.2,20.6,9.7c4.9,6.5,7.4,14.9,7.4,25.1H47.4		c0-8-1.3-14-4-18c-2.7-4-6.5-6-11.4-6c-6.1,0-10.7,3.1-13.7,9.4c-3.1,6.3-4.6,16.9-4.6,31.7c0,14.1,1.9,23.8,5.7,29.1		c3.8,5.3,8.2,8,13.1,8c4.9,0,8.9-1.8,11.7-5.4c2.9-3.6,4.3-9.2,4.3-16.9v-6.3H30.9V49.7H60V100z'
            }
    # 精准医学字体
    elif tp == 'd':
        path = {
            'N':'M60,100H45.6L13.8,27.5h-0.6V100H0V0h14.4l31.7,72.5h0.6V0H60V100z',
            'P':'M60,30.4c0,9.4-2.7,16.8-8,22.2c-5.3,5.5-12.8,8.2-22.3,8.2H13.1V100H0V0h29.7C39.2,0,46.7,2.7,52,8.2		C57.3,13.7,60,21.1,60,30.4z M46.9,30.4c0-7.4-1.7-12.5-5.1-15.2c-3.4-2.7-8.6-4.1-15.4-4.1H13.1v38.6h13.1c6.9,0,12-1.4,15.4-4.1		C45.1,42.9,46.9,37.8,46.9,30.4z',
            'D':'M60,50.3c0,17.9-3.7,30.7-11,38.3C41.6,96.2,31.7,100,19.2,100H0V0h19.2c14,0,24.2,3.9,30.8,11.7		C56.7,19.5,60,32.4,60,50.3z M46.4,50.3c0-14.4-2.3-24.6-6.8-30.4c-4.5-5.8-11.3-8.8-20.4-8.8H13v77.8h6.2c9.1,0,15.8-2.8,20.4-8.5		C44.2,74.8,46.4,64.7,46.4,50.3z',
            'H':'M60,100H46.6V53.2H13.4V100H0V0h13.4v42.1h33.2V0H60V100z',
            'C':'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z'
            }
    # 氨基酸字体
    else:
        path = {
            'A':'M60,100H48l-7-30.6H19L12,100H0L25,0h10L60,100z M38.5,58.4l-8-35.3h-1l-8,35.3H38.5z',
            'R':'M60,100H46.8L27,56.7H12.7V100H0V0h27.5c8.8,0,16,2.2,21.5,6.7c5.5,4.5,8.3,11.8,8.3,21.9		c0,7.8-1.8,13.8-5.5,18.1c-3.7,4.3-7.9,7-12.7,8.2L60,100z M44.6,28.7c0-5.5-1.6-9.7-4.7-12.9c-3.1-3.1-8.2-4.7-15.1-4.7H12.7v34.5		h15.4c4.4,0,8.3-1.3,11.6-3.8C42.9,39.3,44.6,34.9,44.6,28.7z',
            'N':'M60,100H45.6L13.8,27.5h-0.6V100H0V0h14.4l31.7,72.5h0.6V0H60V100z',
            'D':'M60,50.3c0,17.9-3.7,30.7-11,38.3C41.6,96.2,31.7,100,19.2,100H0V0h19.2c14,0,24.2,3.9,30.8,11.7		C56.7,19.5,60,32.4,60,50.3z M46.4,50.3c0-14.4-2.3-24.6-6.8-30.4c-4.5-5.8-11.3-8.8-20.4-8.8H13v77.8h6.2c9.1,0,15.8-2.8,20.4-8.5		C44.2,74.8,46.4,64.7,46.4,50.3z',
            'C':'M60.1,60c-0.4,14.5-3.3,24.8-8.7,30.9c-5.5,6.1-12,9.1-19.7,9.1c-8.7,0-16.2-3.7-22.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,3-31.2,9-40C15,4.4,22.7,0,32.2,0c8,0,14.6,3.1,19.9,9.4c5.3,6.3,7.7,15.1,7.4,26.6h-12		c0-8.4-1.3-14.7-3.8-18.9c-2.6-4.2-6.4-6.3-11.5-6.3c-5.8,0-10.5,3.1-13.9,9.4c-3.5,6.3-5.2,16.9-5.2,31.7		c0,13.7,1.7,23.3,5.2,28.9c3.5,5.5,7.9,8.3,13.4,8.3c4,0,7.6-2,10.9-6c3.3-4,4.9-11.7,4.9-23.1H60.1z',
            'Q':'M60,46.8c0,8.6-0.6,15.9-1.9,21.8c-1.3,5.9-3.3,10.7-6.2,14.2l4.3,8.1l-8.6,9.1l-4.9-9.1		c-1.4,1.1-3.2,1.9-5.4,2.4c-2.2,0.5-4.5,0.8-7,0.8c-9.4,0-16.8-3.7-22.2-11C2.7,75.7,0,63.6,0,46.8c0-16.8,2.7-28.8,8.1-36		C13.5,3.6,20.9,0,30.3,0c9.4,0,16.7,3.6,21.9,10.8C57.4,17.9,60,29.9,60,46.8z M47,46.8c0-15.1-1.6-24.9-4.9-29.6		c-3.2-4.7-7.2-7-11.9-7c-4.7,0-8.7,2.3-12.2,7C14.7,21.9,13,31.7,13,46.8s1.7,25,5.1,29.8c3.4,4.8,7.5,7.3,12.2,7.3		c1.8,0,3.4-0.3,4.9-0.8c1.4-0.5,2.5-1.2,3.2-1.9l-8.6-16.7l7.6-9.1l7.6,14.5c0.7-3.6,1.3-6.8,1.6-9.7C46.8,57.4,47,52.9,47,46.8z',
            'E':'M60,100H0V0h57.1v11.1H13.5v31h40v11.1h-40v35.7H60V100z',
            'G':'M60,100h-9.7l-1.7-8c-1.5,2.3-3.9,4.2-7.1,5.7c-3.2,1.5-6.8,2.3-10.6,2.3c-8,0-15.1-3.7-21.4-11.1		C3.1,81.4,0,69.5,0,53.1c0-17.9,2.9-31.2,8.6-40C14.3,4.4,22.1,0,32,0c8.8,0,15.6,3.2,20.6,9.7c4.9,6.5,7.4,14.9,7.4,25.1H47.4		c0-8-1.3-14-4-18c-2.7-4-6.5-6-11.4-6c-6.1,0-10.7,3.1-13.7,9.4c-3.1,6.3-4.6,16.9-4.6,31.7c0,14.1,1.9,23.8,5.7,29.1		c3.8,5.3,8.2,8,13.1,8c4.9,0,8.9-1.8,11.7-5.4c2.9-3.6,4.3-9.2,4.3-16.9v-6.3H30.9V49.7H60V100z',
            'H':'M60,100H46.6V53.2H13.4V100H0V0h13.4v42.1h33.2V0H60V100z',
            'I':'M23.1,11.5H0V0h60v11.5H36.9v76.9H60V100H0V88.4h23.1V11.5z',
            'L':'M60,100H0V0h13.3v88.9H60V100z',
            'K':'M60,100H45.9L21.1,50.3l-8.6,12.3V100H0V0h12.4v43.3L43.2,0h14.6L29.2,39.2L60,100z',
            'M':'M60,100H48.9V28.7h-1.1L34.4,100h-8.9L12.2,28.7h-1.1V100H0V0h17.2l12.2,66.1h1.1L42.8,0H60V100z',
            'F':'M60,11.1H13.1v32.7h35.4V55H13.1v45H0V0h60V11.1z',
            'P':'M60,30.4c0,9.4-2.7,16.8-8,22.2c-5.3,5.5-12.8,8.2-22.3,8.2H13.1V100H0V0h29.7C39.2,0,46.7,2.7,52,8.2		C57.3,13.7,60,21.1,60,30.4z M46.9,30.4c0-7.4-1.7-12.5-5.1-15.2c-3.4-2.7-8.6-4.1-15.4-4.1H13.1v38.6h13.1c6.9,0,12-1.4,15.4-4.1		C45.1,42.9,46.9,37.8,46.9,30.4z',
            'S':'M60,71.4c0,8.8-2.7,15.7-8.1,20.9c-5.4,5.1-12.8,7.7-22.1,7.7c-9.3,0-16.6-2.7-21.9-8C2.6,86.7,0,80,0,72v-4		h12.9v3.4c0,5.7,1.7,10.1,5,13.1c3.4,3.1,7.3,4.6,11.8,4.6c6,0,10.4-1.6,13.2-4.9c2.8-3.2,4.2-7.1,4.2-11.7c0-3.8-1.7-7.3-5-10.6		c-3.4-3.2-8.2-6.2-14.6-8.9c-9-3.4-15.4-7.2-19.3-11.4c-3.9-4.2-5.9-9.1-5.9-14.9c0-8,2.7-14.5,8.1-19.4C15.8,2.5,22.2,0,29.7,0		c9.7,0,16.7,3,21,8.9c4.3,5.9,6.4,12.3,6.4,19.1H44.3c0.4-4.2-0.8-8-3.4-11.4c-2.6-3.4-6.4-5.1-11.2-5.1c-4.5,0-8,1.2-10.7,3.7		c-2.6,2.5-3.9,5.8-3.9,10c0,3.4,1,6.4,3.1,8.9c2.1,2.5,7.4,5.4,16,8.9c8.2,3.4,14.6,7.5,19.1,12.3C57.8,59.9,60,65.3,60,71.4z',
            'T':'M60,11.1H36.7V100H23.3V11.1H0V0h60V11.1z',
            'W':'M60,0L49.2,100h-9.8l-8.9-74h-1l-8.9,74h-9.8L0,0h10.8l5.4,63h1l7.4-63h10.8l7.4,63h1l5.4-63H60z',
            'Y':'M60,0L35.9,55v45H24.1V55L0,0h11.8l17.9,42.7h0.5L48.2,0H60z',
            'V':'M60,0L35,100H25L0,0h12l17.5,75.7h1L48,0H60z'
            }
    return path

# 颜色定义
def weblogo_color(tp='p'):
    # 碱基颜色
    if tp == 'n':
        color = {
            'A':'#07d794',
            'T':'#786fa6',
            'C':'#e15f41',
            'G':'#3dc1d3',
            }
    # 精准医学颜色
    elif tp == 'd':
        color = {
            'N':'#303952',
            'P':'#574b90',
            'D':'#cf6a87',
            'H':'#546de5',
            'C':'#e15f41'
            }
    # 氨基酸颜色
    else:
        color = {
            'A':'#f5cd79',
            'R':'#3dc1d3',
            'N':'#ea8685',
            'D':'#cf6a87',
            'C':'#778beb',
            'Q':'#f78fb3',
            'E':'#f3a683',
            'G':'#f7d794',
            'H':'#546de5',
            'I':'#e15f41',
            'L':'#f8a5c2',
            'K':'#786fa6',
            'M':'#63cdda',
            'F':'#f19066',
            'P':'#574b90',
            'S':'#e66767',
            'T':'#303952',
            'W':'#778beb',
            'Y':'#cf6a87',
            'V':'#07d794'
            }
    return color

# 配置标尺
def weblogo_ruler(number, dlta, gap, y_ori, label, x_label=None):
    # 总体偏移量
    y = 0 + y_ori
    # 定义xy轴和y轴标签
    out = '<text fill="#333333" x="0" y="10" transform="translate(0,' + str(y+55) + ')rotate(-90, 10, 5)">' + label + '</text><rect x="45" y="' + str(y-10) + '" width="2" height="120" fill="#333333"/><rect x="45" y="' + str(y+108) + '" width="' + str(int(number*24+50)) + '" height="2" fill="#333333"/><rect x="57" y="' + str(y+109) + '" width="2" height="5" fill="#333333"/>'
    # 定义y轴刻度
    for j in range(int(math.log(dlta, 2))+1):
        # 定义y轴数字和y轴大刻度
        out += '<text fill="#333333" x="25" y="' + str(y+114) + '">' + str(j) + '</text><rect x="36" y="' + str(y+108) + '" width="10" height="2" fill="#333333"/>'
        if j < int(math.log(dlta, 2)):
            for i in range(4):
                y -= gap
                # 定义y轴小刻度
                out += '<rect x="41" y="' + str(y+108) + '" width="5" height="2" fill="#333333"/>'
            y -= gap
    # 总体偏移量
    x = 33
    y = 0 + y_ori
    if x_label == None:
        # 定义x轴刻度
        for j in range(int(number/5)+1):
            for i in range(4):
                x += 24
                # 定义x轴小刻度
                out += '<rect x="' + str(x) + '" y="' + str(y+109) + '" width="2" height="5" fill="#333333"/>'
            x += 24
            # 定义x轴数字和x轴大刻度
            out += '<text fill="#333333" x="' + str(x-4) + '" y="' + str(y+134) + '">' + str((j+1)*5) + '</text><rect x="' + str(x) + '" y="' + str(y+109) + '" width="2" height="10" fill="#333333"/>'
    else:
        t_x, t_y = 106, 17
        # 定义x轴刻度
        for j in range(len(x_label)):
            t_x += 7
            t_y -= 17
            x += 24
            # 定义x轴数字和x轴大刻度
            out += '<text fill="#333333" x="' + str(x - 4) + '" y="' + str(y + 134) + '" transform="translate(' + str(t_x) + ',' + str(t_y) + ') rotate(45)">' + x_label[j] + '</text><rect x="' + str(x) + '" y="' + str(y + 109) + '" width="2" height="10" fill="#333333"/>'
    return out

# 删除0值并返回氨基酸索引
def weblogo_del(line, aa_index):
    out1 = []
    out2 = []
    for i in range(len(line)):
        if line[i] != 0:
            out1.append(line[i])
            out2.append(aa_index[i])
    return [out1, out2]

# 更新元素列表
def weblogo_update(line, aa):
    out = [[],[]]
    test = 'yes'
    for i in range(len(line[0])):
        if aa != line[0][i]:
            out[0] = out[0] + [line[0][i]]
            out[1] = out[1] + [line[1][i]]
        elif test != 'yes':
            out[0] = out[0] + [line[0][i]]
            out[1] = out[1] + [line[1][i]]
        else:
            test = 'no'
    return out

# 求取元素上下边界及缩放比例
def weblogo_yc(line, y1, y2, r, gap):
    aa_value = min(line[0])
    aa_id = line[1][line[0].index(aa_value)]
    new_line = weblogo_update(line, aa_value)
    r += aa_value
    y1 -= r*gap*5
    c = (y2-y1) / 100
    return y1, y2, c, new_line, r, aa_id

# 配置元素
def weblogo_logo(value, aa_index, path_index, color, y_ori, gap):
    # 基准坐标
    x = 23
    logo = ''
    for i in range(len(value)):
        # 定义偏移量
        x += 24
        y2 = y_ori + 108
        r = 0
        # 删除0值并返回氨基酸索引
        line = weblogo_del(value[i], aa_index)
        li = len(line[0])
        for j in range(li):
            y1 = y_ori + 108
            if len(line[0]) != 0:
                # 求取元素上下边界及缩放比例
                y1, y2, c, line, r, aa_id = weblogo_yc(line, y1, y2, r, gap)
                y2 = y1
                # 写入元素
                logo += '<path fill="' + color[aa_id] + '" d="' + path_index[aa_id] + '" transform="translate(' + str(x) + ',' + str(y1) + ')scale(0.4,' + str(c) + ')"/>'
    return logo

# 配置标尺像素比例尺
def weblogo_gap(tp='p'):
    if tp == 'p':
        return 5
    elif tp == 'n':
        return 10
    elif tp == 'd':
        return 9

# 绘图
def weblogo_plot(value, label, out=now_path, tp='p', x_label=None):
    # 获取元素索引
    aa_index = visual_create_aa(tp=tp)
    # 获取元素字体
    path_index = weblogo_path(tp=tp)
    # 获取元素颜色
    color = weblogo_color(tp=tp)
    if x_label == None:
        # 设定svg头部
        head = '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(int(len(value[0])*24+100)) + '" height="' + str(len(value)*150) + '">'
    else:
        # 设定svg头部
        head = '<svg xmlns="http://www.w3.org/2000/svg" width="' + str(int(len(value[0]) * 24 + 100)) + '" height="' + str(len(value) * 150+40) + '">'
    # 设定svg尾部
    end = '</svg>'
    # 写入元素
    all_data = ''
    # 基准偏移量
    y_ori = -140
    for i in range(len(value)):
        y_ori += 150
        # 提取数据
        data = value[i]
        # 设置标尺像素比例
        gap = weblogo_gap(tp=tp)
        # 添加标尺
        all_data += weblogo_ruler(len(data), len(data[0]), gap, y_ori, label[i], x_label=x_label)
        # 添加元素
        all_data += weblogo_logo(data, aa_index, path_index, color, y_ori, gap)
    # 汇总svg并输出
    all_data = head + all_data + end
    with open(out, 'w') as f:
        f.write(all_data)

# 配置y轴标签
def weblogo_label(label, li):
    if label == None:
        label = visual_create_n_matrix(x=li, fill='bits')
    else:
        for i in range(li):
            label[i] = label[i] + ' (bits)'
    return label

# main 单图
def weblogo(file=None, data=None, out=now_path, tp='p', x_label=None):
    # 时间戳
    time_id = visual_timestamp()
    if data == None:
        data = weblogo_read(file)
    else:
        data = data
    if out == now_path:
        out = os.path.join(out, time_id + '.svg')
    # 提取信息熵矩阵
    value = weblogo_count(data, tp=tp)
    # 绘图
    weblogo_plot([value], ['bits'], out=out, tp=tp, x_label=x_label)

# main 多图
def weblogo_multy(data=None, label=None, out=now_path, tp='p'):
    # 时间戳
    time_id = visual_timestamp()
    if data == None:
        return data
    else:
        # 配置y轴标签
        label = weblogo_label(label, len(data))
        if out == now_path:
            out = os.path.join(out, time_id + '.svg')
        # 提取信息熵矩阵
        value = []
        for i in data:
            value.append(weblogo_count(i, tp=tp))
        # 绘图
        weblogo_plot(value, label, out=out, tp=tp)
