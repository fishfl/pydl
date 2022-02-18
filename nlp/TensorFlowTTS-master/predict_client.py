import json
import numpy as np
import requests

input_text = "您听到的这段语音是人工合成的"
data = json.dumps({'instances': [input_text]})
print(data)

headers = {"content-type": "application/json"}
json_response = requests.post('http://172.25.146.212:5001/v1/models/tts/versions/1/',
                              data=data, headers=headers)

predictions = json.loads(json_response.text)['predictions']
audios = np.array(predictions)
import soundfile as sf
sf.write('./a.wav', audios, 22050, "PCM_16")