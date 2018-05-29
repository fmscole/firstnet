import urllib,sys
import urllib.request
import json
import numpy as np

host = 'http://stock.market.alicloudapi.com'
path = '/sz-sh-stock-history'
method = 'GET'
appcode = '6a611850fcc942939ec34411c37ffb3d'
querys = 'begin=2015-09-01&code=600004&end=2015-09-02'
bodys = {}
url = host + path + '?' + querys

request = urllib.request.Request(url)
request.add_header('Authorization', 'APPCODE ' + appcode)
response = urllib.request.urlopen(request)
content = response.read()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, bytes):
            return str(obj, encoding='utf-8')
        return json.JSONEncoder.default(self, obj)
        
json_str = json.dumps(content,cls=MyEncoder)

if (content):
    print(content)
    print("                                                                                                  ")
    print(json_str)
