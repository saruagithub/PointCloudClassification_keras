import os
from flask import Flask, request, redirect, url_for, json,make_response,send_from_directory,send_file,app
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import math
from pointtest import provider
import keras
from keras.models import load_model
from flask_cors import *
import xlwt
# import flask_excel as excel


# UPLOAD_FOLDER = 'uploadfiles'
# DOWNLOAD_FOLDER = 'excels'
UPLOAD_FOLDER = 'pointtest/uploadfiles'
DOWNLOAD_FOLDER = 'pointtest/excels'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#upload file type
ALLOWED_EXTENSIONS = set(['h5'])


app = Flask(__name__,static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DOWNLOAD_FOLDER'] = DOWNLOAD_FOLDER
CORS(app, resources=r'/*')

# 默认load model K11
modelName = 'modelK11.h5'
# MODEL = load_model('model/modelK11.h5') #local path
MODEL = load_model('pointtest/model/modelK11.h5') #server path
modelList = []
res = []  # global variable: save all results
pointCloudData = []
pointCloudObject = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone',
      'cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp',
      'laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink',
      'sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']


def allowed_file(filename):
	return '.' in filename and \
		   filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#read file, load data into the model, get the predicted type
def calculate(filename):
	#read data from hdf5 file
	global pointCloudData,MODEL
	predict_data, predict_label = provider.load_h5(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	predict_data = predict_data[:, 0:2048, :]
	pointCloudData = predict_data	# global data
	predict_data = predict_data[:, :, :, np.newaxis]
	predict_label = np.squeeze(predict_label)
	predict_label = keras.utils.to_categorical(predict_label, num_classes=40)
	#use model to predict
	# print("calculate:",modelName)
	pre = MODEL.predict(predict_data, batch_size=32, verbose=1)
	max_probability = 0.0
	index1 = 0 #predicted class
	index2 = 0 #reas class
	accuracyCount = 0 #totall right prediction
	pre_objects = predict_data.shape[0]
	result = [[0 for col in range(3)] for row in range(pre_objects)]
	#calculate the biggest probability and point cloud object type,save into the result list
	for i in range(pre_objects):
		for j in range(40):
			#find the biggest probability,which means the prediction
			if (pre[i][j] > max_probability):
				max_probability = pre[i][j]
				index1 = j
			if (predict_label[i][j] == 1):
				index2 = j
		result[i][0] = index1
		result[i][1] = round(max_probability.astype(float),4) # 4 number after dot
		result[i][2] = index2
		if (index1 == index2):
			accuracyCount += 1
		max_probability = 0 #reset
	if (pre_objects % 10 > 0 ):
		totalpages = (pre_objects // 10) + 1
	else:
		totalpages = pre_objects // 10
	accuracy = round(accuracyCount / predict_data.shape[0],4)
	print(totalpages,accuracy)
	return  totalpages,result,accuracy



@app.route('/', methods=['GET', 'POST'])
def welcome():
	print(BASE_DIR)
	return 'hello!!'

#return models name
@app.route('/getModels', methods=['GET', 'POST'])
def getModels():
	global modelList,modelName
	modelList = [] #init modellist
	for filenames in os.walk("pointtest/model"):
		# 三个参数：返回1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
		for filename in filenames[2]:  # 输出文件信息
			# print("parent is:", parent)
			if (filename.startswith(".")):
				continue
			modelList.append(filename)
	res = {'modelList':modelList,'nowModel':modelName}
	response = make_response(json.dumps(res))
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Methods'] = 'POST'
	response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
	return response

#setModels
@app.route('/setModels', methods=['GET', 'POST'])
def setModels():
	if request.method == 'GET':
		global MODEL,modelName
		modelName = request.args.get('modelname')
		MODEL = load_model('pointtest/model/' + modelName)
		print(modelName)
		response = make_response(json.dumps("ok"))
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'POST'
		response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
		return response


#upload file to predict
@app.route('/upload', methods=['GET', 'POST', 'OPTIONS'])
def upload_file():
	if request.method == 'POST':
		file = request.files['filename']
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			global res
			totalPages,res,accu = calculate(filename)
			res0 = res[0:10]
			# print(res0)
			dic_res = {'totalPages':totalPages,'result':res0,'accracy':accu}
		# redirect(url_for('upload',filename=filename))
		# redirect(url_for('upload_file'))
		#跨域问题
		response = make_response(json.dumps(dic_res))
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'POST'
		response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
		return response

#get the current page result data
@app.route('/pagedata', methods=['GET'])
def pageDatas():
	pageIndex = int(request.args.get('pageIndex'))
	global res
	res1 = res[(pageIndex-1)*10:(pageIndex)*10]
	response = make_response(json.dumps(res1))
	response.headers['Access-Control-Allow-Origin'] = '*'
	response.headers['Access-Control-Allow-Methods'] = 'GET'
	response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
	return response

#export excel
@app.route('/generateExcel', methods=['GET'])
def generateExcel():
	global modelName
	if request.method == "GET":
		excelName = request.args.get('nowModel').split('.')[0] + '_' + request.args.get('filename') + '.xls'
		print(excelName)
		# write result list into the excel file
		excelfile = xlwt.Workbook(encoding='utf-8')
		table = excelfile.add_sheet('predict1')
		table.write(0, 0, 'No.')  # 写入数据table.write(行,列,value)
		table.write(0, 1, '预测概率')
		table.write(0, 2, '预测类型ID')
		table.write(0, 3, '预测类型')
		table.write(0, 4, '实际类型ID')
		table.write(0, 5, '实际类型')
		table.write(0, 6, '预测正误')
		global res,pointCloudObject
		for i in range(len(res)):
			table.write(i + 1, 0, i + 1)
			table.write(i + 1, 1, res[i][1])
			table.write(i + 1, 2, res[i][0])
			table.write(i + 1, 3, pointCloudObject[res[i][0]])
			table.write(i + 1, 4, res[i][2])
			table.write(i + 1, 5, pointCloudObject[res[i][2]])
			if (res[i][0] == res[i][2]):
				table.write(i + 1, 6, 1)
			else:
				table.write(i + 1, 6, 0)
		excelfile.save(os.path.join(DOWNLOAD_FOLDER, excelName))
		response =  make_response(send_from_directory('/Users/wangxue/gitpro/DL/pointcloud/pointtest/excels',excelName,as_attachment=True))
		# response =  make_response(app.send_static_file(os.path.join(DOWNLOAD_FOLDER, excelName)))
		# response =  send_from_directory(app.config['DOWNLOAD_FOLDER'],excelName,as_attachment=True)
		response.headers["Content-Disposition"] = 'attachment; filename={}'.format(excelName.encode().decode('latin-1'))
		response.headers["Content-Type"] = 'application/octet-stream'
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'GET'
		response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
		return response

#draw point cloud data
@app.route('/drawPoint', methods=['GET'])
def drawPoint():
	if request.method == "GET":
		pointCloudId = int(request.args.get('id'))
		global pointCloudData
		print("id",pointCloudId,"shape",pointCloudData.shape)
		pointCloudDataList = pointCloudData[pointCloudId].tolist() #numpy array transfer to list
		# 跨域问题
		response = make_response(json.dumps(pointCloudDataList))
		response.headers['Access-Control-Allow-Origin'] = '*'
		response.headers['Access-Control-Allow-Methods'] = 'GET'
		response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
		return response


if __name__ == '__main__':
	app.run('127.0.0.1', 5001)
