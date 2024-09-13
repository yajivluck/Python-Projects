import sys
import socket
import time

from Agent import Agent
from BoardRule import BoardRule


COLOR_LEN = 30
DEF_PORT = 12345
DEF_MSG_SIZE = 100
BUF_SIZE = 10000

def send(s, b):
    
    print(f"Sending to server: {b}")
    s.send(b.encode())
    #Returns sent move
    return b
        
    
def receive(s,sent_move):
    data = s.recv(BUF_SIZE)
    data = data.decode()
    
    while data == sent_move:
        
        #print('server acknowledged you sent',data)
        
        data = s.recv(BUF_SIZE)
        data = data.decode()
        
    data = data.rstrip('\n')


    print(f"Received from server: {data}")
    return data
        
        
def main():
    serverIP = None
    serverPort = DEF_PORT
    color = "white"
    game_id = 'game37'

    for i in range(1, len(sys.argv)):
        if sys.argv[i] == '-c':
            color = sys.argv[i + 1]
        elif sys.argv[i] == '-p':
            serverPort = int(sys.argv[i + 1])
        elif sys.argv[i] == '-g':
            game_id = sys.argv[i + 1]
        else:
            serverIP = sys.argv[i]

    if serverIP is None:
        print("Usage: python script.py [-c color] [-p port] serverhost")
        sys.exit(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        sock.connect((serverIP, serverPort))
    except socket.error as e:
        print(f"Error connecting to {serverIP} on port {serverPort} (TCP): {e}")
        sys.exit(1)

    buf = f"{game_id} {color}\n"
    #buf = f"mytestgame {color}\n"
    first_acknowledge = send(sock, buf)
    #Accept to receive the first game settings back (client side as acknowldeged)
    first_receive = receive(sock, sent_move = "")
    
    
    #TODO Instantiate board configuration from here 

    
    #board = get_from_file... and add to arguments of Board
    Board = BoardRule()
    
    #Print starting board configuration
    print(Board)

    
    
        
    #First check who sends first
    if color == 'white':
        
        agent_white = Agent(board_state = Board, symbol = 'O', depth = 3)
        
        white_move = agent_white.choose_move()
        #Make move on Board
        
                
        white_move = send(sock,white_move + '\n')
        Board.make_move(move = white_move)
        print(Board)


        while True:
            
            black_move = receive(sock,sent_move = white_move)
            
            print(black_move)
            Board.make_move(move = black_move)
            print(Board)

            time.sleep(1)
            white_move = agent_white.choose_move()

            white_move = send(sock, white_move + '\n')
            Board.make_move(move = white_move)
            print(Board)


        
    
    else:
         
        
        agent_black = Agent(board_state = Board, symbol = 'X', depth = 3)
        
        white_move = receive(sock,sent_move = '')
        Board.make_move(move = white_move)
        print(Board)


        
        while True:
            
            black_move = agent_black.choose_move()
                        
            black_move = send(sock, black_move + '\n')
            Board.make_move(move = black_move)
            print(Board)

            time.sleep(1)
            white_move = receive(sock,sent_move = black_move)
            if Board.make_move(move = white_move): print(Board) 
            
            else: 
                white_move = receive(sock,sent_move = black_move)
                print(Board)


            
            
            
            
if __name__ == "__main__":
    main()

    
   


    
    
    
