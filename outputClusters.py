import numpy as np
import base64
from io import BytesIO
from PIL import Image

def generatePage(clusterMap, imageMetadata, clusterMeans, pageName, extraData=None, n=20):
  html = "<!doctype html><html><head><title>" + pageName + "</title></head>"
  html += "<body><h1>" + pageName + "</h1>"

  if extraData != None:
    html += "<p>" + str(extraData) + "</p>"

  html += "<table>"
  for i in clusterMap:
    html += '<tr><td><h2>' + str(i) + "</h2></td>"

    pilImg = Image.fromarray(clusterMeans[i], mode="L")
    buff = BytesIO()
    pilImg.save(buff, format="JPEG")
    medianB64 = base64.b64encode(buff.getvalue()).decode("utf-8")
    html += '<td><img src="data:img/jpg;base64,' + medianB64 + '" /></td>'

    for index in clusterMap[i][:n]:
      metadata = imageMetadata[index]
      w, h = metadata['dimensions']
      filename = metadata['filename']
      height = 200.
      width = 200. * h / w
      html += '<td><img src="photos/' + filename + '" width="' + str(width) + '" height="200" /></td>'
    html += "</tr>"
  html += "</table>"

  with open("output/" + pageName + ".html", "w") as f:
    f.write(html)

