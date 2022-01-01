
[TrainingTime,TrainingAccuracy,elm_model]=elm_train_m(train_x,train_y,1, 90, 'sigmoid',1)
% elm_model90=elm_model
%save D:\exp\elm\elm_model90 elm_model
test_x=[test_y test_x];
