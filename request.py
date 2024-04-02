import requests

url = 'http://localhost:5000/upload_image'
files = {'image': open('../Task 1/Flask/test2.jpg', 'rb')}
response = requests.post(url, files=files)

# print(response.json())
