# -*- coding: utf-8 -*-

import numpy as np
import cv2

from interpretartela import definir_objetos_tela
from lertecla import  ler_tecla_pressionada

from directkeys import PressKey, ReleaseKey

from tensorflow1 import NeuralNetwork
from decision_tree import DecisionTree

def main():
    
    print("Fit pong screen into the window")
    print("Press 'up' or 'down' to start infering actions.")
    print("Press 'q' for quit.")
    
    infering = False
    
    #Load neural network
    nn = NeuralNetwork()
    nn.load()
        
    #Call function to clear buffer of pressed keys
    ler_tecla_pressionada()
    
    last_pos_h = 0
    last_pos_v = 0
    
    #Keeping getting track of the object locations and keys pressed
    while True:

        screen, obj_locations = definir_objetos_tela()
        
        key_pressed = ler_tecla_pressionada()
        
        cv2.imshow("Pong-Py", screen)
    
        #Pass next frame every 10ms
        #Exit when 'q' is pressed
        if cv2.waitKey(1) == ord('q') or key_pressed == -1:
            cv2.destroyAllWindows()
            break
            
        #Calculate speed
        h_speed = obj_locations[0] - last_pos_h
        v_speed = obj_locations[1] - last_pos_v

        last_pos_h = obj_locations[0]
        last_pos_v = obj_locations[1]

        screen_features = np.insert(obj_locations, 2, [h_speed, v_speed])
        
        #Check whether we are already saving data
        if infering:
                        
            prediction_probs = nn.predict([screen_features])[0] 
            prediction = np.argmax(prediction_probs)
            
            if prediction == 0:
                print(prediction_probs, "Nothing")
                ReleaseKey(0x48)
                ReleaseKey(0x50)
            elif prediction == 1:
                print(prediction_probs, "Up")
                ReleaseKey(0x50)
                PressKey(0x48)
            elif prediction == 2:
                print(prediction_probs, "Down")
                ReleaseKey(0x48)
                PressKey(0x50)
        
        elif key_pressed > 0:
            print("Infering")
            infering = True
                        
if __name__ == "__main__":
    main()
