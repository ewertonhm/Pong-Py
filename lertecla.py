# Aqui definimos a função para capturar as teclas pressionadas. 
# Precisamos da lib numpy e da lib cv2(opencv). Vamos utilizar o módulo win32api que faz parte do pacote pywin32. 
# Ela utiliza o método GetAsyncKeyState para verificar cada tecla que está pressionada. 
# O código 38 é a seta para cima do teclado, o código 40 é a seta para baixo, e a função ord(“Q”) 
# retorna o código da letra Q. Queremos verificar apenas essas teclas pois usamos a tecla Q para finalizar o programa 
# e as setas para movimentar as barras. Utilizamos a comparação down_pressed==up_pressed porque caso as duas teclas 
# tenham o mesmo estado, soltas ou pressionadas, a barra não se movimentará, então interpretamos como um caso só.

import win32api as wapi

def ler_tecla_pressionada():
    #Como pong só vai para cima e para baixo, essas são as unicas teclas verificadas
    #Codigo para tecla para cima é 38 e para baixo é 40

    up_pressed = wapi.GetAsyncKeyState(38)
    down_pressed = wapi.GetAsyncKeyState(40)
    exit_pressed = wapi.GetAsyncKeyState(ord("Q"))
    
    #se 'q' for pressionado retorna -1
    if exit_pressed:
        return -1
    
    #se nem uma tecla estiver pressionada (up e down estarão com status igual), 
    # ou se ambas estiverem pressionadas (up e down também estarão com status igual), retonar 0
    if down_pressed == up_pressed:
        return 0
    
    if up_pressed:
        return 1
    
    if down_pressed:
        return 2