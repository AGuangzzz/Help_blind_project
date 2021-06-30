from aip import AipOcr
import cv2
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

class Ocr_module():
    def __init__(self,):
        APP_ID = "22729886"
        API_KEY = "EDrxKsEyjZ6VZXPMc9yZrnlp"
        SECRET_KEY = "Ax5HgTPasDqpLocaMIFQEjGWI6bQWkdH"
        # APP_ID = "16889798"
        # API_KEY = "TTP9lDBpD4vvrxoaBWyq7wn1"
        # SECRET_KEY = "zNFwdQTdmIpa4YSnIA73GgFk1zjjbI5W"
        self.client_for_ocr = AipOcr(APP_ID, API_KEY, SECRET_KEY)

    def get_ocr(self, image):
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        cv2.imwrite("ocr_work/ocr_photo.jpg", image)
        
        # 读取图像
        image = get_file_content("ocr_work/ocr_photo.jpg")
        options = {"language_type": "CHN_ENG", "detect_direction": "true", "detect_language": "true"}
        # 进行检测
        result_dict = self.client_for_ocr.basicAccurate(image, options)

        # 获得预测结果
        result_list = result_dict["words_result"]
        orc_save_sentence = ""
        for item in result_list:
            for string in item:
                orc_save_sentence = orc_save_sentence + item[string]
        return orc_save_sentence