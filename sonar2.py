import numpy as np
#pandas doc file csv
import pandas as pd
from sklearn.model_selection import train_test_split
#thuoc thuat toan phan lop, phan biet min va da 
from sklearn.linear_model import LogisticRegression
#kiem tra do chinh xac ma tran diem vs accuracy
from sklearn.metrics import accuracy_score

from pandas import read_csv
#load du lieu dataset vao, header = none de cai header mac dinh cua file csv, neu de =1
# thi dong dau tien se la header ten dau de cua file csv,
sonar_data = read_csv('D:/CNN/sonar.all-data.csv', header=None)

# # in cot thu 3 trong csv ra
# hades = sonar_data[3]
# hades.to_csv('D:/CNN/jena.csv')


#hien thi du lieu dataset
sonar_data.head()
#print(sonar_data.head())
# dem so hang va cot
sonar_data.shape
#print(sonar_data.shape())

#thong ke du lieu mo ta du lieu trung binh
sonar_data.describe() 
#lay du lieu cot mo ta phan biet vat la min hay la da, o day cot phan biet la 60, dem xem 
#co bao nhieu da va min o day min la 111 da la 97
sonar_data[60].value_counts()
#thong ke du lieu trung binh , xac xuat xay ra doi voi tung loai cua tung, 111 min va 97 da 
#cua tung so lieu du lieu cua 1 vat the
sonar_data.groupby(60).mean()

# tach du lieu va dan nhan de thuan tien cho qua trinh train hay input dau vao, tao mang cho 
# tao mang 60 cot dau doi vs X, doi voi Y du lieu phan biet min da cuoi cung la Y
X = sonar_data.drop(columns=60, axis=1)
Y = sonar_data[60]
#print(X)
#print(Y)

#Training and Test data
#validation va seed phu thuoc cach chon sao cho phu hop nhat
#stratifi phan tang Y vi du lieu cua Y la du lieu phan biet
# va random lay du lieu phu hop va cu the sao cho phu hop
validation_size = 0.2
seed = 1
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size,stratify=Y, random_state=seed)

#.shape la tinh so luong cot can chia 
# print(X.shape, X_train.shape, X_test.shape)

# print(X_train)
# print(Y_train)

#Model Training --> Logistic Regression
model = LogisticRegression()


#fit mo hinh
model.fit(X_train, Y_train)

#Model Evaluation

#accuracy tao bien
#model.predict du doan mo hinh cho du lieu de tren, model.predict la du doan du lieu
X_train_prediction = model.predict(X_train)
#train chinh xac du lieu chuong chinh vs diem chinh xac cua du lieu da train ben tren va du lieu Y train
#bang cac so sanh du lieu X predic va du lieu y train
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 

print('Accuracy on training data : ', training_data_accuracy)

#accuracy du lieu kiem tra
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) 

print('Accuracy on test data : ', test_data_accuracy)

# #kiem tra du lieu xem du doan dung bao nhieu phan tram
#print(model.score(X_test, Y_test))

#Du doan ve muc tieu du lieu co san
#lay du lieu hang thu bao nhieu roi tru di 1
# row hang du lieu nhap vao
row = 198
input_data = X.iloc[row-1]


# chuyen ve mang numpy tao bien vecto dau vao
input_data_as_numpy_array = np.asarray(input_data)
# dinh hinh lai mang numpy
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# tao bien mo hinh de du doan
Y_predict = model.predict(input_data_reshaped)

print(Y_predict)

if (Y_predict[0]=='R'):
    print('The object is a Rock')
else:
    print('The object is a mine')

# #du lieu nhap ngoai

# X_new = (0.0034,0.0069,0.0172,0.0334,0.0317,0.0281,0.0395,0.0277,0.0323,0.0459,0.0490,0.0992,0.1425,0.1196,0.0628,0.0907,0.1177,0.1429,0.1223,0.1104,0.1847,0.3715,0.4382,0.5707,0.6654,0.7476,0.7654,0.8555,0.9720,0.9221,0.7502,0.7209,0.7757,0.6055,0.5021,0.4499,0.3947,0.4281,0.4427,0.3749,0.1972,0.0511,0.0793,0.1269,0.1533,0.0690,0.0402,0.0534,0.0228,0.0073,0.0069,0.0062,0.0127,0.0052,0.0057,0.0093,0.0044,0.0009,0.0056,0.0038,)

# # chuyen ve mang numpy tao bien vecto dau vao
# X_new_as_numpy_array = np.asarray(X_new)
# # dinh hinh lai mang numpy
# X_new_reshaped = X_new_as_numpy_array.reshape(1,-1)
# # tao bien mo hinh de du doan
# Y_predict = model.predict(X_new_reshaped)

# print(Y_predict)

# if (Y_predict[0]=='R'):
#     print('The object is a Rock')
# else:
#     print('The object is a mine')