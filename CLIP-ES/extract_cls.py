
with open('map_clsloc.txt', 'r') as f:
    lines = f.readlines()  # 读取文件中的所有行

result = []
for line in lines:
    elements = line.split()  # 将行切割成元素列表
    element = elements[2].replace('_', ' ')  # 选择第三个元素并替换下划线为空格
    result.append("'" + element + "', ")  # 将处理后的元素添加到结果列表中

result_str = ''.join(result)[:-2]  # 将结果列表转换为字符串并删除最后的逗号和空格
print(result_str)  # 打印结果字符串
