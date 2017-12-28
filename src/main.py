import numpy as np
import random
import math
import pickle

board = np.zeros([8, 8], dtype = np.int16)
dx = [0, -1, -1, -1, 0, 1, 1, 1]
dy = [1, 1, 0, -1, -1, -1, 0, 1]
white = 1
black = -1
empty = 0
Q = {}
init_value = [
	[3, 0, 2, 1, 1, 1, 0, 3],
	[0, 2, 1, 2, 2, 1, 2, 0],
	[2, 1, 2, 1, 1, 2, 1, 2],
	[1, 2, 1, 0, 0, 1, 2, 1],
	[1, 2, 1, 0, 0, 1, 2, 1],
	[2, 1, 2, 1, 1, 2, 1, 2],
	[0, 2, 1, 2, 2, 1, 2, 0],
	[3, 0, 2, 1, 1, 1, 0, 3]
]

def init():
	global board
	board = np.zeros([8, 8], dtype = np.int16)
	board[3][3] = white
	board[4][4] = white
	board[3][4] = black
	board[4][3] = black

def load_Q():
	global Q
	f = open('Q.pkl', 'rb')
	Q = pickle.load(f)
	f.close()
	print len(Q)

def put_stone(x, y, color):
	for d in range(8):
		xx = x + dx[d]
		yy = y + dy[d]
		while (in_board(xx, yy) and board[xx][yy] == -color):
			xx += dx[d]
			yy += dy[d]
		if (in_board(xx, yy) and board[xx][yy] == color):
			board[x][y] = color
			xx = x + dx[d]
			yy = y + dy[d]
			while(board[xx][yy] == -color):
				board[xx][yy] = color
				xx += dx[d]
				yy += dy[d]

def in_board(x, y):
	return x>=0 and x<8 and y>=0 and y<8

def available(x, y, color):
	if (not in_board(x, y)):
		return False
	if (board[x][y] != 0):
		return False
	for d in range(8):
		xx = x + dx[d]
		yy = y + dy[d]
		cnt = 0
		while (in_board(xx, yy) and board[xx][yy] == -color):
			xx += dx[d]
			yy += dy[d]
			cnt += 1
		if (cnt and in_board(xx, yy) and board[xx][yy] == color):
			return True
	return False

def count():
	b = 0
	w = 0
	for i in range(8):
		for j in range(8):
			if(board[i][j] == black):
				b += 1
			elif(board[i][j] == white):
				w += 1
	return (w, b)

def isend():
	for i in range(8):
		for j in range(8):
			if(available(i, j, white) or available(i, j, black)):
				return False
	return count()

def board2dint():
	i1 = 0L
	i2 = 0L
	for i in range(4):
		for j in range(8):
			i1 |= (board[i][j]&3L) << ((i * 8 + j) * 2)
	for i in range(4, 8):
		for j in range(8):
			i2 |= (board[i][j]&3L) << (((i-4) * 8 + j) * 2)
	return (i1, i2)

def dint2board(dint):
	for i in range(4):
		for j in range(8):
			color = (dint[0] >> ((i * 8 + j) * 2)) & 3
			if(color == 0):
				board[i][j] = empty
			elif(color == 1):
				board[i][j] = white
			elif(color == 3):
				board[i][j] = black
			else:
				return False
	for i in range(4, 8):
		for j in range(8):
			color = (dint[1] >> (((i-4) * 8 + j) * 2)) & 3
			if(color == 0):
				board[i][j] = empty
			elif(color == 1):
				board[i][j] = white
			elif(color == 3):
				board[i][j] = black
			else:
				return False
	return True

def getQ(pos, color, _board = ()):
	#pos = (x, y)
	if(len(_board) == 0):
		_board = board2dint()
	key = (_board, pos, color)
	val = Q.get(key)
	if(val == None):
		return init_value[pos[0]][pos[1]]
	return val

def setQ(pos, val, color, _board = ()):
	if(len(_board) == 0):
		_board = board2dint()
	key = (_board, pos, color)
	Q[key] = val

def train(lr, fr, award, eps = 0.01, stpcrtr = 1e-3, maxiter = 1e4):
	#learning rate, factor rate, epsilon for eps-greedy
	start_board = board2dint()
	i = 0
	while(i < maxiter):
		dint2board(start_board)
		end = False
		cc = black
		nor = 0.0
		#eps *= 0.8
		cnt = 0
		while(not end):
			avail = []
			pos = ()
			for j in range(8):
				for k in range(8):
					if(available(j, k, cc)):
						avail.append((getQ((j,k), cc), (j, k)))
			avail.sort(key=lambda act: act[0], reverse=True)
			if(len(avail) == 0):
				cc = -cc
				continue
			if(random.random() < eps):
				pos = avail[random.randint(0, len(avail) - 1)][1]
			else:
				pos = avail[0][1]
			Q0 = getQ(pos,cc)
			b0 = board2dint()
			put_stone(pos[0], pos[1], cc)
			cc = - cc
			nextQ = -10.0
			judge = isend()
			if(judge != False):
				nextQ = float(judge[(cc+1)/2] - judge[1-(cc+1)/2])/64 * 2 - 1
				end = True
			else:
				for j in range(8):
					for k in range(8):
						if(available(j, k, cc)):
							nextQ = max(nextQ, getQ((j, k), cc))
			newQ = Q0 + lr * (award + fr * (-nextQ) - Q0)
			nor += (newQ - Q0) * (newQ - Q0)
			setQ(pos, newQ, -cc, b0)
			cnt += 1
		i += 1
		print i, cnt, nor

def train_step(iters):
	for i in range(iters):
		init()
		train(0.1, 0.1, 0.1, maxiter = 10, eps = 0.05)
		print len(Q)
		f1 = file('Q.pkl', 'wb')
		pickle.dump(Q, f1, True) 
		f1.close()
		print "file saved"

def show_board():
	print "0 1 2 3 4 5 6 7 Y/X"
	for i in range(8):
		s = ""
		for j in range(8):
			if (board[i][j] == black):
				s += "B "
			elif (board[i][j] == white):
				s += "W "
			else:
				s += "  "
		print "%s%d" % (s, i)

def gameplay():
	init()
	print "please input the color"
	color = raw_input()
	if (color == "black"):
		pc = black
	else:
		pc = white
	cc = black
	judge = isend()
	while (judge == False):
		show_board()
		if (cc == pc):
			x = -1
			y = -1
			while (not available(x, y, pc)):
				x = input("please input X\n")
				y = input("please input Y\n")
			put_stone(x, y, pc)
			cc = -cc
		else:
			avail = []
			pos = ()
			for j in range(8):
				for k in range(8):
					if(available(j, k, cc)):
						avail.append((getQ((j,k), cc), (j, k)))
			avail.sort(key=lambda act: act[0], reverse=True)
			if(len(avail) == 0):
				cc = -cc
				continue
			pos = avail[0][1]
			print pos
			put_stone(pos[0], pos[1], cc)
			cc = -cc
		judge = isend()
	print "white %d : black %d" % (judge[0], judge[1])

def main():
	load_Q()
	#train_step(4000)
	gameplay()

if __name__ == "__main__":
	main()