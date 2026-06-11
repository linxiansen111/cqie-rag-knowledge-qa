"""
意图识别模块
基于关键词+规则匹配的简易意图识别器
"""

from typing import Dict, List, Optional
import re


class IntentRecognizer:
    """意图识别器：根据用户问题识别意图类别"""

    # 意图类别定义及对应的关键词规则
    INTENT_RULES: Dict[str, List[str]] = {
        "school_info": [
            r"学校(简介|概况|介绍|历史|背景|规模|面积|校区|位置|在哪)",
            r"(成立于|创建于|创办于|建校|前身|更名|升格)",
            r"南泉|双桥|校区",
            r"校训|办学|理念|定位|特色",
            r"几个学院|院系设置|二级学院|下设.*学院",
        ],
        "major_query": [
            r"专业|学科|课程|本科|学士|学位",
            r"计算机|软件|物联网|电子|通信|自动化",
            r"数字媒体|动画|视觉传达|设计",
            r"电子商务|市场营销|财务管理",
            r"土木工程|工程造价|工程管理",
            r"数据科学|大数据|人工智能|AI",
            r"开设.*专业|哪些专业|有什么专业",
        ],
        "admission": [
            r"招生|录取|报考|志愿|投档",
            r"分数线|录取线|多少分|排名",
            r"学费|收费|费用|多少钱",
            r"学制|几年|就读",
            r"代码|12608",
            r"招生办|录取规则|分数优先",
        ],
        "campus_life": [
            r"宿舍|寝室|住宿|住.*条件|几人间",
            r"食堂|餐饮|吃饭|生活费|饭菜",
            r"图书馆|藏书|自习|阅览",
            r"运动|体育|操场|篮球|健身",
            r"校园(环境|设施|生活)",
            r"社团|活动|学生组织",
            r"WiFi|网络|空调|热水器|卫生间",
        ],
        "contact": [
            r"电话|手机|联系方式|联系.*方式",
            r"地址|邮编|在哪|怎么(去|走|到)",
            r"邮箱|邮件|电子邮箱|Email",
            r"网站|官网|网址|官方网站",
            r"教务处|学生处|总机",
            r"怎么联系|如何联系|联系学校",
        ],
    }

    # 意图到中文标签的映射
    INTENT_LABELS: Dict[str, str] = {
        "school_info": "学校概况",
        "major_query": "专业查询",
        "admission": "招生信息",
        "campus_life": "校园生活",
        "contact": "联系方式",
        "general": "通用问答",
    }

    def __init__(self):
        """初始化意图识别器"""
        # 编译所有正则表达式
        self._compiled_rules: Dict[str, List[re.Pattern]] = {}
        for intent, patterns in self.INTENT_RULES.items():
            self._compiled_rules[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]

    def recognize(self, question: str, context: Optional[List[Dict]] = None) -> str:
        """
        识别用户问题的意图

        Args:
            question: 用户问题文本
            context: 可选的上文对话历史，用于上下文关联

        Returns:
            意图标签: school_info, major_query, admission, campus_life, contact, general
        """
        if not question or not question.strip():
            return "general"

        question = question.strip()

        # 如果有上下文且问题偏短（可能是追问），使用上文的意图
        # 阈值放宽到25字，覆盖更多追问场景（如"计算机专业录取分数线多少？"）
        if context and len(question) < 25:
            follow_up_patterns = [
                r"^(这|那|它|他|她|该|其|这个|那个|他们|它们|她们)(的|里|些|个|种)",
                r"^(最大|最小|最多|最少|最好|最差|第一|最后|另外|其他|别的)",
                r"^(有(没有|几个|哪些)|是(不是|否)|能不能|可不可以|怎么样)",
                r"^(多少|哪个|哪些|谁|什么|何时|哪里)",
                r"^(然后|还有|那|那么|所以|再说)",
            ]
            is_follow_up = any(re.match(p, question) for p in follow_up_patterns)
            if is_follow_up and context:
                # 返回上文的意图
                for msg in reversed(context):
                    if "intent" in msg and msg["intent"] != "general":
                        return msg["intent"]

        # 对每个意图类别进行关键词匹配
        scores: Dict[str, int] = {}
        for intent, patterns in self._compiled_rules.items():
            score = 0
            for pattern in patterns:
                matches = pattern.findall(question)
                score += len(matches)
            if score > 0:
                scores[intent] = score

        if not scores:
            return "general"

        # 返回匹配得分最高的意图
        best_intent = max(scores, key=scores.get)
        return best_intent

    def get_intent_label(self, intent: str) -> str:
        """获取意图的中文标签"""
        return self.INTENT_LABELS.get(intent, "通用问答")

    def get_all_intents(self) -> Dict[str, str]:
        """获取所有意图类别及其中文标签"""
        return dict(self.INTENT_LABELS)
