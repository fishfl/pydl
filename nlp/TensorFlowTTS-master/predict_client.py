import json
import numpy as np
import requests

input_text = "你听到的这段语音是人工合成的"
data = json.dumps({'instances': [input_text]})
print(data)

headers = {"content-type": "application/json",
           "user_id": "746478570813272064",
           "token": "f0d77297-9e10-4eb9-853a-2fdb420a368c",
           "model_name": "tts-offical"}

json_response = requests.post('http://aimodelmarket.cn:8080/v1/models/tts/versions/1/',
                              data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
print(predictions)

audios = np.array(predictions)
import soundfile as sf
sf.write('./a.wav', audios, 22050, "PCM_16")