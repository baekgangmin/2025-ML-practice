import warnings 
warnings.simplefilter("ignore")
import preprocess
import svm 
import graph_plot
from svm import train_linear_model, train_poly_model, train_rbf_model, train_sig_model, evaluate_model

def main():
    # 데이터를 불러옵니다.
    x_train, y_train, x_val, y_val = preprocess.load_data('./data/iris.csv', independent_var = ['petal length', 'sepal length'], response_var = 'class_num')
    
    # <ToDo>: svm.py 안의 함수를 사용해 네 SVM 모델을 불러오고 학습시킵니다.
    svm_linear = train_linear_model(x_train, y_train)
    svm_poly = train_poly_model(x_train, y_train)
    svm_rbf = train_rbf_model(x_train, y_train)
    svm_sig = train_sig_model(x_train, y_train)
    
    print("Independent variable: {}".format("petal length" + ' and ' + 'sepal length'))
    
    # SVM 모델의 그래프를 그립니다.
    graph_plot.svm_model_plot(svm_linear, svm_poly, svm_rbf, svm_sig, x_train, y_train, feature_name = ['petal length', 'sepal length'] )
    
    # <ToDo>: svm.py 안의 함수를 사용해, 학습된 네 모델의 성능을 정확도로써 측정합니다.
    mean_acc_linear = evaluate_model(svm_linear, x_val, y_val)
    mean_acc_poly = evaluate_model(svm_poly, x_val, y_val)
    mean_acc_rbf = evaluate_model(svm_rbf, x_val, y_val)
    mean_acc_sig = evaluate_model(svm_sig, x_val, y_val)
    
    print("mean accuracy for linear kernel: {}%".format(mean_acc_linear*100))
    print("mean accuracy for poly kernel: {}%".format(mean_acc_poly*100))
    print("mean accuracy for rbf kernel: {}%".format(mean_acc_rbf*100))
    print("mean accuracy for sigmoid kernel: {}%".format(mean_acc_sig*100))
    
if __name__ == "__main__":
    main()