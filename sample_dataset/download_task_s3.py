# import scaleapi
# import requests
import boto3
from urllib.parse import urlparse
# import project_labels
# 1 TASK
# Init #
s3 = boto3.client('s3')
scale_api_key = 'live_c51c4273f60f4bcb9e86578c372aa51d'
project = 'object_labeling_3d'
# client = scaleapi.ScaleClient(scale_api_key)
# raw_data = client.tasks(project=project, status='completed')[:1]
# task = raw_data[0]
# Get Labels #
# label_url = task.response["annotations"]["url"]
# labels = requests.get(label_url)
# open('task_labels.json', 'wb').write(labels.content)
# Get Point Cloud Data #
# pcd_list = task.params["attachments"]


jsons = ["s3://autoronto-3d/1598992517/lidar/1598992510763825000.json",
      "s3://autoronto-3d/1598992517/lidar/1598992510971290000.json"]
      # "s3://autoronto-3d/1598992517/lidar/1598992511178703000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992511386161000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992511593531000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992511800909000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992512008190000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992512215521000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992512422880000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992512630426000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992512837880000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992513045340000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992513252789000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992513460128000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992513667556000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992513874895000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992514082229000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992514289704000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992514497140000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992514704536000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992514912021000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992515119491000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992515326943000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992515534438000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992515741872000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992515949198000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992516156559000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992516363922000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992516571324000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992516778791000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992516986242000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992517193715000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992517401164000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992517608523000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992517815902000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992518023351000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992518230690000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992518438118000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992518645461000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992518852918000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992519060294000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992519267751000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992519475101000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992519682543000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992519889926000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992520097410000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992520304780000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992520512247000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992520719659000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992520927033000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992521134463000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992521341816000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992521549292000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992521756613000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992521964066000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992522171493000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992522378985000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992522586381000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992522793833000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992523001249000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992523208601000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992523415878000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992523623314000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992523830590000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992524037887000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992524245312000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992524452717000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992524660162000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992524867688000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992525075237000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992525282686000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992525490125000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992525697571000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992525904805000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992526112081000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992526319389000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992526526785000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992526734197000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992526941727000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992527149151000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992527356672000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992527564167000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992527771689000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992527979156000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992528186479000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992528393772000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992528601029000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992528808131000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992529015394000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992529222837000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992529430356000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992529637948000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992529845569000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992530053121000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992530260554000.json",
      # "s3://autoronto-3d/1598992517/lidar/1598992530468023000.json"]


j = 0
for i in jsons:
    print(i)
    url = urlparse(i)
    bucket = url.netloc
    key = url.path.lstrip('/')
    s3.download_file(bucket, key, 'point_cloud_data/task_point_cloud{}.json'.format(j))
    # project_labels.visualize_bounding_boxes('point_cloud_data/task_point_cloud{}.json'.format(j), 'labels.json', j)
    # j += 1