import json
from draft import requests


#发送代码
def json_send(dataPModel,url):
    headers = {"Content-type": "application/json", "Accept": "text/plain", "charset": "UTF-8"}
    response = requests.post(url=url, headers=headers, data=json.dumps(dataPModel))
    response_text = response.text
    return json.loads(response_text)

if __name__ == "__main__":
    #---------------------------#
    #   声音测试
    #---------------------------#
    url = 'http://192.168.8.100:8502/play_voice'
    dataPModel = {"sentence": "请说出地点名称"}
    result = json_send(dataPModel, url)