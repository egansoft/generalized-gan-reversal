import numpy as np
import base64

def generatePage(clusterMap, imageMetadata, pageName, extraData=None, n=20):
  html = "<!doctype html><html><head><title>" + pageName + "</title></head>"
  html += "<body><h1>" + pageName + "</h1>"

  if extraData != None:
    html += "<p>" + str(extraData) + "</p>"

  html += "<table>"
  for i in clusterMap:
    html += '<tr><td><h2>' + str(i) + "</h2></td>"
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

