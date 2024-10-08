import re


def contains_chinese(text):
    """
    检查给定的文本是否包含中文字符。

    :param text: 要检查的字符串
    :return: 如果包含中文字符则返回 True，否则返回 False
    """
    # 匹配中文字符的正则表达式
    pattern = re.compile(r'[\u4e00-\u9fff]+')

    # 如果找到匹配项，则字符串包含中文
    if pattern.search(text):
        return True
    else:
        return False


# 测试函数
def test_contains_chinese():
    # 测试用例
    test_cases = [
        ("Hello, 世界!", True),
        ("This is English.", False),
        ("你好，世界！", True),
        ("12345", False),
        ("こんにちは", False),  # 日语
        ("Hello世界", True),
        ("", False)
    ]

    for text, expected in test_cases:
        result = contains_chinese(text)
        print(f"Text: '{text}', Contains Chinese: {result}, Expected: {expected}")
        assert result == expected, f"Error: '{text}' should return {expected}"

    print("All tests passed!")


# 运行测试
if __name__ == "__main__":
    test_contains_chinese()