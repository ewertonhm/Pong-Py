from lertela import ler_tela             
import cv2                      
import numpy as np


#--------------------------------------------------------------------------------------------

def determina_local_objetos(img):
	# Essa função recebe a imagem da tela do jogo em escalas de cinza e retorna 6 valores dentro de um vetor: 
	# A posição horizontal e vertical da bola, a posição horizontal e vertical da primeira barra e a posição horizontal e 
	# vertical da segunda barra (que controlamos).
    try:
        
        #Get connected compontents in the image
        n_elem, labels, stats, centroids = cv2.connectedComponentsWithStats(img, 8, cv2.CV_32S)

        #Calculate features to determine which elements are ball and bars 
        calc_features = list()    
        for i, (x0,y0,width,height,area) in enumerate(stats):
            #Calc ball feature
            ball_feature = abs(width/height - 1 + area - width*height)
            #Bar feature
            bar_feature = abs(height/width - 4.6 + area - width*height)

            calc_features.append((i, ball_feature, bar_feature))


        #Sort values to get the most probable indexes of ball and bars
        ball_ind = sorted(calc_features, key=lambda a: a[1])[0][0]
        bars_ind = [bar_data[0] for bar_data in sorted(calc_features, key=lambda a: a[2])[0:2]]


        #Get the centroids with the indexes
        ball_center = centroids[ball_ind]
        bars_center = centroids[bars_ind]

        #Get left bar and right bar based on sorted value of the x position
        sorted_bars = sorted(bars_center, key=lambda a: a[0])

        left_bar_cent, right_bar_cent = sorted_bars[0], sorted_bars[1]

        return np.array([ball_center, left_bar_cent, right_bar_cent]).reshape(-1)
    
    except:
        return np.array([0,0,0,0,0,0])


#--------------------------------------------------------------------------------------------

def pegar_tela():
	# A função pegar_tela retorna uma região da tela da tela em escalas de cinza. 
	# Primeiro a função ler_tela retorna a região definida pelos pontos x1=200, y1=200, x2=720+200 e y2=405+200
	# Colocamos os valores finais em forma de soma (720+200) devido a facilidade de modificar o tamanho da região capturada 
	# e seu offset, no caso a região de tamanho 720x405 com offset de 200 na horizontal e 200 na vertical será retornada. 
	# A função cvtColor converte as cores da imagem capturada para escalas de cinza.
    #screen = ler_tela(region=(200,200,480+200,270+200))
    screen = ler_tela(region=(200,200,720+200,405+200))
    #screen = ler_tela(region=(200,200,960+200,540+200))
    #print(screen.shape)
    #screen = cv2.resize(screen, (480,270))

    screen = cv2.cvtColor(screen, cv2.COLOR_RGB2GRAY)

    return screen

#--------------------------------------------------------------------------------------------



def definir_objetos_tela():
    # Aqui combinamos as funções pegar_tela e determina_local_objetos 
    # para mostrar a localização dos objetos na tela com círculos cinzas para fins de debug.

    screen = pegar_tela()
    
    obj_locations = determina_local_objetos(screen)

    #Determina o centro 
    ball_center = tuple(np.round(obj_locations[:2]).astype(int))
    bar1_center = tuple(np.round(obj_locations[2:4]).astype(int))
    bar2_center = tuple(np.round(obj_locations[4:6]).astype(int))

    #Desenha circulos na localização dos objetos
    cv2.circle(screen, ball_center, 5, 128, -1)
    cv2.circle(screen, bar1_center, 5, 128, -1)
    cv2.circle(screen, bar2_center, 5, 128, -1)
    
    return screen, obj_locations

#--------------------------------------------------------------------------------------------

#Por fim, ao executar o arquivo, capturamos a tela e as informações dos objetos do jogo; 
# Em seguida fazemos uma stream do conteúdo da imagem capturada em uma tela auxiliar até que o programa seja fechado 
# ou a tecla ‘q’ for pressionada.

if __name__ == "__main__":
    print("Fit pong screen into the window:")

    while True:

        screen, obj_locations = determina_local_objetos()

        cv2.imshow("Pong-Py", screen)
    
        #Exibe um frame a cada 25ms
        #Para quando a tecla 'q' é precionada
        if cv2.waitKey(25) == ord('q'):
            cv2.destroyAllWindows()
            break
