import tensorflow as tf
import numpy as np
from collections import Counter 
from sklearn.model_selection import train_test_split

# tensorflow é a biblioteca usada para criar e treinar as neural networks
# numpy é usada para tratar vetores e matrizes
# Counter é uma classe utilizada para contar objetos
# train_test_split é uma função importada da biblioteca scikit-learn para misturar e separar nossos dados em dados de treino e dados de teste

# -------------------------------------------------------------------------------------------

# Aqui no construtor do objeto criamos nossa rede neural. Importante notar que temos 6 features:
# 	1 Posição horizontal da bola
# 	2 Posição vertical da bola
# 	3 Posição horizontal da barra 1
# 	4 Posição vertical da barra 1
# 	5 Posição horizontal da barra 1
# 	6 Posição vertical da barra 1
# Porém, no código está definido que temos 8, isso é devido pelo fato que temos que adicionar mais duas features 
# para que a rede aprenda corretamente: a velocidade horizontal e vertical da bola.

class NeuralNetwork():
    
    def __init__(self):
                
        tf.reset_default_graph()

        n_features = 8

        self.x = tf.placeholder(dtype=tf.float32, shape=[None, n_features])    
        self.y = tf.placeholder(dtype=tf.int32, shape=[None])
        y_onehot = tf.one_hot(self.y, depth=3)
    
        h1 = tf.layers.dense(self.x, 20, activation=tf.nn.relu)
        h2 = tf.layers.dense(h1, 20, activation=tf.nn.relu)
        #h3 = tf.layers.dense(h2, 10, activation=tf.nn.relu)
    
        self.logits = tf.layers.dense(h2, 3, activation=None)

        sc = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y_onehot)

        self.cost = tf.reduce_mean(sc)

        # Accuracy
        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(y_onehot, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        #Optimizer
        self.optimizer = tf.train.AdamOptimizer(0.01).minimize(self.cost)
        #self.optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(self.cost)

# Quando captamos os dados, preenchemos o valor corresponde as teclas apertadas com 0 (nenhuma tecla), 
# 1 (tecla cima) e 2 (tecla baixo). Como precisamos de variáveis categoricas, fazemos isso com a função tf.one_hot.
# Poderíamos criar as conexões da rede no “braço”, multiplicando e adicionando matrizes, 
# mas ao invés disso utilizamos a função tf.layer.dense que já faz tudo isso por nós, bastando passar os parâmetros 
# de cada camada da rede. Utilizamos a camada de entrada com 8 unidades, 2 intermediárias com 20 unidades cada uma e a 
# camada de saída com 3 unidades. Embora não tenha feito muitos experimentos, esse tamanho se mostrou suficiente para 
# bons resultados.

# Criamos um tensor para calcular o erro (cost) com cross-entropy para utilizarmos para otimizar a rede e outro de 
# performance (accuracy) para medir o desempenho. O algoritmo de otimização Adam se mostrou melhor que o tradicional 
# Gradient Descent então o utilizamos com um learning rate de 0.01 para treinar a rede.

    def fit(self, X, Y):
        
        #Split train, test and validation set
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
        x_valid, x_test, y_valid, y_test = train_test_split(x_test, y_test, test_size=0.5, random_state=42)        
        
        epochs = 25000

        self.sess = tf.Session()
        
        # Initializing the variables
        self.sess.run(tf.global_variables_initializer())

        
        for e in range(epochs):
        
            #Run optimizer and compute cost                
            cost_value, _ = self.sess.run([self.cost, self.optimizer], feed_dict={
                self.x: x_train,
                self.y: y_train
            })
            
            if e % 500 == 0:
                print("Epoch: {} Cost: {}".format(e, cost_value))

            #Run accuracy and compute its value        
            acc_value = self.sess.run(self.accuracy, feed_dict={
                self.x: x_valid,
                self.y: y_valid
            })

            if e % 500 == 0:
                print("Accuracy: {}".format(acc_value))
                print("")
                
                
        #Calculate final accuracy    
        final_acc = self.sess.run(self.accuracy, feed_dict={
            self.x: x_test,
            self.y: y_test
        })
    
        print("Final accuracy: {}".format(final_acc))

# Criamos a função fit para treinarmos nossa neural network. 
# Utilizamos a função train_test_split para separamos os inputs e outputs coletados em inputs/ouputs de treino 
# (para ensinarmos a rede) e inputs/outputs de teste (para medirmos a performance da rede). 
# Iremos utilizar 25000 epochs, isso é, iremos treinar nossa rede com os mesmo dados de treino 25 mil vezes. 
# Criamos a nossa sessão para executar as operações com tf.Session e iniciamos todas as variáveis declaradas 
# (basicamente os weights e biases criados por tf.layers.dense) com a função tf.global_variables_initializer. 
# Executamos as operações de otimização da rede (optimizer) e cálculo de erro (cost) com os dados de treino 
# ao mesmo tempo que medimos a performance da rede com a operação accuracy com os dados de validação. 
# A cada 500 epochs, exibimos a situação atual da rede (performance e erro). 
# Por fim, verificamos a performance da rede com a mesma operação accuracy mas dessa vez com os dados de teste.        

    def predict(self, X):
        
        prediction = self.sess.run(self.logits, feed_dict={
            self.x: X
        })
        
        return prediction
    
    def save(self):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, "./model/model_v15.ckpt")
    
    def load(self):
        self.sess = tf.Session()
        
        saver = tf.train.Saver()
        saver.restore(self.sess, "./model/model_v15.ckpt")

# Aqui temos três funções:

# predict: Utilizamos para inferência. Executa a operação logits na rede treinada (utilizando a mesma sessão) 
# para retornar a ação que deve ser tomada para o conjunto de dados X passado.

# save: Salva o estado atual da sessão da rede (utilizando a classe tf.train.Saver), 
# salvando várias informações, inclusive o valor dos pesos (weights) que foram calculados quando fizemos o treinamento. 
# Utilizamos isso para não ter que treinar a rede toda vez que o projeto for fechado e aberto.

# load: Carrega uma rede neural previamente salva utilizando a função save.

def set_ball_speed(dataset):
    """
    Function to get ball speed based on the difference of horizontal and vertical positions in subsequent frames.
    """
    
    #Since we don't know the previous position of the first sample, we will have one less datapoint
    #Velocity(T) = Position(T) - Position(T-1)
    speed_datapoints = dataset[1:, :2] - dataset[0:-1, :2]
    #Insert new datapoints after the ball position features
    new_dataset = np.concatenate((dataset[1:, :2], speed_datapoints, dataset[1:, 2:]), axis=1)
    return new_dataset

# Aqui utilizamos a função set_ball_speed para inserir no dataset a velocidade da bola em cada instante. 

# Para qual direção a bola circulada está indo? Para cima ou para baixo? Para direita ou para a esquerda?

# Como saber se devemos subir ou não nossa barra para defender? Se a bola estiver vindo em nossa direção, devemos subir,
# caso contrário podemos ficar parados. Logo precisamos saber para onde a bola está se movimentando,
# isto é sua atual variação no espaço, informação conhecida como velocidade.

# Como cada ponto dos dados de treino contém a posição atual da bola,
# basta para cada frame verificarmos a posição anterior da bola e subtrair da atual:
# speed_datapoints = dataset[1:,2] — dataset[0:-1,2] .
# Como para o primeiro frame não temos a posição anterior,
# começamos a contar a partir do segundo datapoint e teremos um datapoint a menos, o que não influencia em nada.
# Por fim, inserimos os dados calculados em novo dataset e retornamos ele.

if __name__ == "__main__":
    
    #Load dataset
    dataset = np.load(file="traindata_v15.npy", encoding='bytes')

    #Inspect dataset
    labels_counter = Counter(dataset[:,6].tolist())
    print(labels_counter)
    print("Prediction must be higher than: {}".format(labels_counter[0.0]/dataset.shape[0]))
    
    dataset = set_ball_speed(dataset)
    
    neural_network = NeuralNetwork()
    
    X_data = dataset[:, :-1]
    Y_data = dataset[:, -1]

    neural_network.fit(X_data, Y_data)
    neural_network.save()

# Chegando ao final do arquivo, carregamos os dados de treino gravados no post anterior e verificamos a frequência
# de cada label do dataset, isso é, quantas vezes apertamos para cima,
# para baixo ou ficamos parados nos dados de treino gravados. Isso é importante pois ficamos muito mais tempo
# parados do que movimentando a barra, logo se ficarmos 70% do tempo parado e nossa rede tiver um desempenho de 70%,
# existem grandes chances dela ter decidido simplesmente ficar parada o tempo todo, então o ideal é que tenhamos um
# desempenho maior que 70%. Calculamos então a velocidade da bola com set_ball_speed, separamos os dados de treino
# em inputs e output e finalmente treinamos e salvamos nossa rede para uso posterior utilizando as funções comentadas
# anteriormente.
