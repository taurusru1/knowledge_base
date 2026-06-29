# utils/json_utils.py

import json
from datetime import datetime, date
from bson import ObjectId


class MongoJSONEncoder(json.JSONEncoder):
    """
    自定义 JSON 编码器，支持 MongoDB ObjectId 和 datetime 等非标类型。
    """

    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        return super().default(obj)


def json_dumps(obj, **kwargs):
    """
    json.dumps 封装，自动处理 ObjectId 等类型。

    :param obj: 要序列化的对象
    :param kwargs: 传给 json.dumps 的其他参数
    :return: JSON 字符串
    """
    return json.dumps(obj, cls=MongoJSONEncoder, **kwargs)
