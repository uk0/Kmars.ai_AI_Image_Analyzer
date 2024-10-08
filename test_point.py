import xml.etree.ElementTree as ET
import re
from typing import List, Dict


def parse_points(description: str) -> List[Dict[str, any]]:
    """
    解析描述中的point和points标签，提取所有坐标对和其他信息。

    :param description: 包含point或points标签的描述字符串
    :return: 包含解析后信息的字典列表
    """
    xml_string = f"<root>{description}</root>"

    try:
        root = ET.fromstring(xml_string)
    except ET.ParseError:
        return []

    points_data = []

    for point in root.findall('.//point') + root.findall('.//points'):
        point_info = {
            'coordinates': [],
            'alt': point.get('alt', ''),
            'text': point.text.strip() if point.text else ''
        }

        attrs = point.attrib
        coord_pairs = []

        # 首先检查是否有 x 和 y 属性
        if 'x' in attrs and 'y' in attrs:
            coord_pairs.append((float(attrs['x']), float(attrs['y'])))

        # 然后检查是否有 x1, y1, x2, y2 等格式
        i = 1
        while f'x{i}' in attrs and f'y{i}' in attrs:
            coord_pairs.append((float(attrs[f'x{i}']), float(attrs[f'y{i}'])))
            i += 1

        # 如果仍然没有找到坐标，尝试解析属性值中的所有数字对
        if not coord_pairs:
            all_values = ' '.join(attrs.values())
            coord_pairs = [(float(x), float(y)) for x, y in re.findall(r'(\d+\.?\d*)\s+(\d+\.?\d*)', all_values)]

        point_info['coordinates'] = coord_pairs
        points_data.append(point_info)

    return points_data

if __name__ == '__main__':
    # 测试用例
    test_cases = [
        ('''<points x1="14.0" y1="73.2" x2="16.7" y2="63.6" x3="96.5" y3="59.0" alt="红色盖子的物品">红色 cover</points>''', [{'coordinates': [(10.0, 20.0), (30.0, 40.0)], 'alt': '', 'text': 'Some text here'}]),
        ('''<point x="55.4" y="19.0" x1="14.0"  alt="white cover of the switch cover">white cover of the switch cover</point>''', [{'coordinates': [(10.0, 20.0), (30.0, 40.0)], 'alt': '', 'text': 'Some text here'}]),
        ('''<point x="55.4" y="19.0" x1="14.0" y1="73.2" x2="16.7" y2="63.6" x3="96.5" y3="59.0"  alt="white cover of the switch cover">white cover of the switch cover</point>''', [{'coordinates': [(10.0, 20.0), (30.0, 40.0)], 'alt': '', 'text': 'Some text here'}]),

    ]

    for description, expected in test_cases:
        result = parse_points(description)
        # print(f"Description: '{description}', Parsed Points: {result}, Expected: {expected}")
        print(f" Parsed Points: {result},")
