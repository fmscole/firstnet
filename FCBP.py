import numpy
import scipy.special
import matplotlib.pyplot
class neuralNetwork :
    #initialise the neural network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate) :
        # set number of nodes in each input, hidden, output layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # link weight matrices, wih and who
        # weights inside the arrays are w_i_j, where link is from
        # node i to node j in the next layer
        # w11 w21
        # w12 w22 etc
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, 0.5), 
                                       (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, 0.5), 
                                       (self.onodes, self.hnodes))
        self.read();
        # learning rate
        self.lr = learningrate
        # activation function is the sigmoid function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.n=1
        self.m=1
        pass
    # train the neural network
    def train(self, inputs_list, targets_list) :
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)

        # output layer error is the (target actual)
        output_errors = targets - final_outputs
        t=targets.argmax()==final_outputs.argmax()
        if t: self.m=self.m+1
        self.n=self.n+1
        if self.n%10000==0:
            cc=self.m/10000
            if cc>0.9:self.lr=0.01
            print(self.n,cc,numpy.sqrt(numpy.sum(output_errors**2)/10))
            self.m=0
        # hidden layer error is the output_errors, split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        # update the weights for the links between the hidden and output layers
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), 
                                        numpy.transpose(hidden_outputs))
        # update the weights for the links between the input and hidden layers
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), 
                                        numpy.transpose(inputs))
        #numpy.savetxt(r"C:/Z/wh.csv",self.who,fmt='%.2f')
        
        pass
    # query the neural network
    def query(self, inputs_list) :
        # convert inputs list to 2d array
        inputs = numpy.array(inputs_list, ndmin=2).T
        # calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih, inputs)
        # calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)
        # calculate signals into final output layer
        final_inputs = numpy.dot(self.who, hidden_outputs)
        # calculate the signals emerging from final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
    def save(self):
        numpy.savetxt(r"C:/Z/who.csv",self.who,fmt='%.2f')
        numpy.savetxt(r"C:/Z/wih.csv",self.wih,fmt='%.2f')
        pass
    def read(self):
        #self.who = numpy.loadtxt(r"C:/Z/who.csv")
        #self.wih = numpy.loadtxt(r"C:/Z/wih.csv")

        #print(self.who)
        pass
input_nodes = 784
hidden_nodes = 200
output_nodes = 10

# learning rate
learning_rate = 0.1

# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

data_file = open(r"C:\搜狗高速下载\mnist_train.csv", 'r')  #一个可读的文件对象
# 因为文件不大，所以一次读了整个文件。理论上应该一行一行的读。
training_data_list = data_file .readlines()  # 返回一个数据列表，data_list[i]表示第i样本
data_file.close()

#print(len(training_data_list))

#all_values = training_data_list[7].split(',')
#image_array = numpy.array(all_values[1:], dtype=float).reshape((28,28))
#print(all_values[0:1])
#matplotlib.pyplot.imshow(image_array)#, cmap='Greys', interpolation='None')
#matplotlib.pyplot.show()

epochs = 1

for e in range(epochs):
    # go through all records in the training data set
    print(e,"_____________________________________________________________________________________________________")
    for record in training_data_list[1:60000]:
        # split the record by the ',' commas
        all_values = record.split(',')
        # scale and shift the inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # create the target output values (all 0.01, except the desired label which is 0.99)
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0] is the target label for this record
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)

        pass

n.save()
# 加载测试集
# load the mnist test data CSV file into a list
test_data_file = open(r"C:\搜狗高速下载\mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()
#print(len(test_data_list))
# 测试神经网络
#test_data_list=training_data_list
# test the neural network

# scorecard for how well the network performs, initially empty
scorecard = []

# go through all the records in the test data set
for record in test_data_list[1:100]:
    # split the record by the ',' commas
    all_values = record.split(',')
    # correct answer is first value
    correct_label = int(all_values[0])
    # scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    # query the network
    outputs = n.query(inputs)
    # the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    # append correct or incorrect to list
    if (label == correct_label):
        # network's answer matches correct answer, add 1 to scorecard
        scorecard.append(1)
    else:
        # network's answer doesn't match correct answer, add 0 to scorecard
        scorecard.append(0)
        pass

    pass
a=numpy.asarray(scorecard)

print(a.sum())
print(a.size)
print(a.sum()/a.size)


