import cv2
import numpy as np

import base64

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img

data_uri = "data:image/png;base64,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KAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQTEPrrWVqJAAECBAgQIECAQJMCQn+TY1EUAQIECBAgQIAAgXoCQn89SysRIECAAAECBAgQaFJA6G9yLIoiQIAAAQIECBAgUE9A6K9naSUCBAgQIECAAAECTQoI/U2ORVEECBAgQIAAAQIE6gkI/fUsrUSAAAECBAgQIECgSQGhv8mxKIoAAQIECBAgQIBAPQGhv56llQgQIECAAAECBAg0KSD0NzkWRREgQIAAAQIECBCoJyD017O0EgECBAgQIECAAIEmBYT+JseiKAIECBAgQIAAAQL1BIT+epZWIkCAAAECBAgQINCkgNDf5FgURYAAAQIECBAgQKCegNBfz9JKBAgQIECAAAECBJoUEPqbHIuiCBAgQIAAAQIECNQT+P+RvQBQuiqz8QAAAABJRU5ErkJggg=="
img = readb64(data_uri)
cv2.imshow("asd",img)


