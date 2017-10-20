# A rotina desse arquivo é utilizada para gravar os movimentos do jogador e o estado dos objetos do jogo a cada instante. 
# Utilizamos a cada instante a função interpretartela para capturar a tela do jogo e seus atributos e a 
# função lertecla para saber qual tecla está sendo pressionada neste momento. 
# Antes de começar gravar qualquer ação, a rotina permite que o usuário ajuste a tela do jogo dentro do 
# range da tela auxiliar, já que só será capturado o que estiver dentro desta tela auxiliar. Após isso, 
# ao pressionar as teclas para cima ou para baixo, a flag saving_data setará e a rotina de gravação entrará em ação. 
# Juntamos os dados da tela e da tecla pressionada no mesmo vetor train_data_point e após isso adicionamos essa variável 
# no buffer train_data_buffer. Para terminarmos a gravação e salvar os dados em um arquivo, apenas pressionamos a tecla Q.
# Os dados de treino serão salvos em um arquivo .npy. Sem segredos.


import numpy as np
import cv2

from interpretartela import definir_objetos_tela
from lertecla import ler_tecla_pressionada

def main():
    
    print("Ajuste o tamanho da tela")
    print("Pressione para cima ou para baixo para iniciar a gravação de dados.")
    print("Pressione q para salvar e sair")
    
    saving_data = False

    train_data_buffer = list()
    
    #chama a função para zerar o leitor de teclas pressionadas
    ler_tecla_pressionada()
    
    #Fica lendo a posição dos objetos na tela e das teclas pressionadas 
    while True:

        screen, obj_locations = definir_objetos_tela()
        
        key_pressed = ler_tecla_pressionada()
        
        #Exibe a tela Auxiliar
        cv2.imshow("Pong-Py", screen)
    
        #Atualiza o frame a cada 10ms
        #Para quando a tecla 'q' é pressionada
        if cv2.waitKey(1) == ord('q') or key_pressed == -1:
            cv2.destroyAllWindows()
            break
        
        #Verifica se estamos ou não salvando dados
        if saving_data:

            #Junta dados da tela com a tecla pressionada naquele momento.
            train_data_point = np.append(obj_locations, key_pressed)

            #salva no buffer de dados
            train_data_buffer.append(train_data_point)
            
            print("Datapoint: {}".format(len(train_data_buffer)))
        
        elif key_pressed > 0:
            saving_data = True       
    
    #Converte a lista de dados em um numpy array
    train_data_buffer = np.array(train_data_buffer)
    
    print(train_data_buffer.shape)
    print(train_data_buffer[:100])

    np.save("traindata_v15.npy", train_data_buffer)
    print('Dados Salvos.')                    

if __name__ == "__main__":
    main()