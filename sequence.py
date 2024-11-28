from layer import Layer
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
class sequence :
    def __init__(self , num_of_layers ,epoch,l_r,activation,bias):
        self.layers = []
        self.num_of_layers = num_of_layers
        self.data  =None
        self.l_r=l_r
        self.activation=activation
        self.bias=bias

        self.epoch = epoch
    def build_layers(self, layers):
        input_size = 5  # Default input size; replace with actual dataset features if dynamic
        for i in range(self.num_of_layers):
            if i == 0:
                self.layers.append(Layer(layers[i], self.activation, self.l_r, input_size))
            else:
                self.layers.append(Layer(layers[i], self.activation, self.l_r, layers[i-1]))

        # Add final output layer with 3 classes
        self.layers.append(Layer(3, self.activation, self.l_r, layers[-1]))



    
        




 
                

    def back_propagation(self, target, sample):
       

        self.layers[-1].error = (target - self.layers[-1].a_out) * self.layers[-1].differentiating

        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].error = np.dot(self.layers[i+1].error, self.layers[i+1].W) * self.layers[i].differentiating

        self.layers[0].W += self.layers[0].learning_rate * np.outer(self.layers[0].error.astype(np.float64), sample.astype(np.float64))
        if self.bias:
         self.layers[0].bias += self.layers[0].learning_rate * self.layers[0].error

        for i in range(1, len(self.layers)):
            self.layers[i].W += self.layers[i].learning_rate * np.outer(self.layers[i].error, self.layers[i-1].a_out)

            if self.bias: 
             self.layers[i].bias += self.layers[i].learning_rate * self.layers[i].error

    




   
        







   

    def forward_propagation(self, sample):
        self.layers[0].a_out=np.dot(sample,self.layers[0].W.T)
        if self.bias:
            self.layers[0].a_out+=self.layers[0].bias
           
        self.layers[0].a_out=self.layers[0].a_out = [
                self.layers[0].activation(x, h) for h, x in enumerate(self.layers[0].a_out)]
        for i in range(1,len(self.layers)):
            self.layers[i].a_out=np.dot(self.layers[i-1].a_out,self.layers[i].W.T)
            if self.bias:
             self.layers[i].a_out+=self.layers[i].bias

            self.layers[i].a_out = [
                self.layers[i].activation(x, h) for h, x in enumerate(self.layers[i].a_out)]




         
     
                





    def preprocess(self):
      self.data=pd.read_csv('birds.csv')
      gender_distribution = self.data.groupby('bird category')['gender'].apply(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
      self.data['gender'] = self.data.apply(lambda row: gender_distribution[row['bird category']] if pd.isnull(row['gender']) else row['gender'], axis=1) 
      label_encoder=preprocessing.LabelEncoder()
      self.data.iloc[:,0]=label_encoder.fit_transform(self.data.iloc[:,0])


      hot_encoder=preprocessing.OneHotEncoder(sparse_output=False)
      encoded_columns = hot_encoder.fit_transform(self.data.iloc[:,-1].values.reshape(-1,1))
      encoded_df = pd.DataFrame(encoded_columns, columns=hot_encoder.get_feature_names_out([self.data.columns[-1]]))
      self.data = pd.concat([self.data.drop(self.data.columns[-1], axis=1), encoded_df], axis=1)
      normalizer=preprocessing.MinMaxScaler()

      self.data.iloc[:,1:5]=normalizer.fit_transform(self.data.iloc[:,1:5])
      print(self.data)
      X=self.data.iloc[:,0:5]
      Y=self.data.iloc[:,5:]

      x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.4,stratify=Y,shuffle=True)
      return x_train,x_test,y_train,y_test
    


    def train(self):
        x_train,x_test,y_train,y_test=self.preprocess()
        x_train = x_train.to_numpy()
        x_test = x_test.to_numpy()
        y_train=y_train.to_numpy()
        y_test=y_test.to_numpy()


        for j in range(self.epoch):
            for i,sample in enumerate(x_train):
                self.forward_propagation(sample)
                self.back_propagation(y_train[i],sample)
        acc_test,predicted1=self.test(x_test,y_test)
        acc_train,predicted2=self.test(x_train,y_train)
        conf_mat=self.confusion_mat(predicted1,y_test)
        return acc_test,acc_train,conf_mat





    def test(self,x_test,y_test):
        correct_predictions=0
        predicted=[0]
        for i, sample in enumerate(x_test):
            self.forward_propagation(sample)
            predicted_output = self.layers[-1].a_out
            true_output = y_test[i]


            predicted_class = np.argmax(predicted_output)
            predicted.append(predicted_class)

            true_class = np.argmax(true_output)

            if predicted_class == true_class:
                correct_predictions += 1

        accuracy = correct_predictions / len(x_test) * 100

       

        return  accuracy,predicted


    def confusion_mat(self,pred,real):
      conf_mat=[[0, 0, 0], [0, 0, 0], [0, 0, 0]]

      for i in range(len(real)):
         predicted_class = pred[i]
         true_class = np.argmax(real[i])

         conf_mat[true_class][predicted_class]+=1
                      

      return conf_mat
      






