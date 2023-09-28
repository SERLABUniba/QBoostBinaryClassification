import binascii
import os
import socket
import pickle
import argparse
from time import sleep

# Definisci l'indirizzo IP e la porta del server
HOST = '127.0.0.1'
PORT = 514



def sends_json(sock, json_data):
    try:
        # Invia il JSON serializzato usando pickle
        #data = binascii.b2a_hex(os.urandom(8))
        #id = "admin"
        #sourceIP = "127.0.0.1"
        #message = '{"UUID": "' + id +'", "UUID_ECU": "f1b186f8fc", "vehicleModel": "TestVehicleModel", "eventID": "send payload", "eventCategory": "flow", "sourceIP": "' + sourceIP + '", "Timestamp": "1478198376.389427", "ID CAN": "00a0", "DLC": 8, "DATA CAN": "' + data + '"}'
        for message in json_data:
            sock.sendto(pickle.dumps(message), (HOST, PORT))
        
        print("JSON inviato con successo.")
        sleep(5)
    except Exception:
        print("NON VA")

def main():
    parser = argparse.ArgumentParser(
        description='Invia un JSON a un server remoto tramite socket e pickle')
    parser.add_argument('-jf', type=str, help='Il percorso del file JSON da inviare')
    
    args = parser.parse_args()

    #try:
    
    # Carica il contenuto del file JSON
    with open(args.jf, 'r') as json_file:
        json_data = json_file.readlines()
    
    # Invia il JSON al server
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sends_json(sock, json_data)


    #except Exception:
    #    print("Errore generale")

if __name__ == '__main__':
    main()
