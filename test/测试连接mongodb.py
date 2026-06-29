import pymongo

myclient = pymongo.MongoClient("mongodb://192.168.150.101:27017/")
mydb = myclient["user"]
mycol = mydb["collection"]

# 插入数据
# mydict = { "name": "RUNOOB", "alexa": "10000", "url": "https://www.runoob.com" }
#
# x = mycol.insert_one(mydict)
# print(x)
# print(x)

# ins = { "name": "uuu", "alexa": "250", "url": "https://www.pppp.com" }
# mycol.insert_one(ins)

# 查询数据
for x in mycol.find():
    print(x)
