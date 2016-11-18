import numpy as np
import math

# one filter
# one channel - NEXT: extend to 2 or 3 channels!!!
# two images (P = 2)

m = 2
r = 3
alpha = m + r - 1 # alpha x alpha

H = 8 # this is the height of the input WITHOUT PADDING, SO same height as output
W = 8
K = 2
C = 3
N = 2
P = N * math.ceil(H/m) * math.ceil(W/m)

G = np.array([
    [1, 0, 0],
    [0.5, 0.5, 0.5],
    [0.5, -0.5, 0.5],
    [0, 0, 1]
    ])

B = np.array([
    [1, 0, 0, 0],
    [0, 1, -1, 1],
    [-1, 1, 1, 0],
    [0, 0, 0, -1]
    ])

A = np.array([
    [1, 0],
    [1, 1],
    [1, -1],
    [0, -1]
    ])

# two images with two channels each
# indexed N then C
D = np.array([
    [
        [
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10]
        ],
        [
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10]
        ],
        [
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10]
        ]
    ],
    [
        [
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10]
        ],
        [
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10]
        ],
        [
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10],
            [1,2,3,4,5,6,7,8,9,10]
        ]
    ]
])

# filters, indexed K then C
g = np.array([
    [
        [
            [1,1,1],
            [1,1,1],
            [1,1,1]
        ],
        [
            [2,2,2],
            [2,2,2],
            [2,2,2]
        ],
        [
            [3,3,3],
            [3,3,3],
            [3,3,3]
        ],
    ],
    [
        [
            [4,4,4],
            [4,4,4],
            [4,4,4]
        ],
        [
            [5,5,5],
            [5,5,5],
            [5,5,5]
        ],
        [
            [6,6,6],
            [6,6,6],
            [6,6,6]
        ]
    ]
])

# how to split up D appropriately to access tiles
# could make very redundant data where each "image" is just a tile
# that would allow me to keep existing code below
# better: each time we multiply, get appropriate block of image
# slicing creates copy, in C we can have temp buffer to copy over?
# need to write output indices one by one since Y won't be cut up into tile outputs

U = np.empty(shape=(alpha, alpha, K, C))
for k in range(K):
    for c in range(C):
        u = G.dot(g[k][c]).dot(G.T)
        # scatter
        for xi in range(alpha):
            for nu in range(alpha):
                U[xi][nu][k][c] = u[xi][nu]

V = np.empty(shape=(alpha, alpha, C, P))
for i in range(N):
    for c in range(C):
        # for y in range(0, H+2-m, m):
        #     for x in range(0, W+2-m, m):
        for y in range(math.ceil(H/m)):
            for x in range(math.ceil(W/m)):
                print(y, x)
                d = D[i,c,y*m:y*m+alpha,x*m:x*m+alpha]
                #v = B.T.dot(D[b][c]).dot(B) #c, b here is swapped, how to get tile out
                v = B.T.dot(d).dot(B)
                #b = i * (number of tiles per image) + y * (number of tiles per row) + x
                b = i * (math.ceil(H/m) * math.ceil(W/m)) + y * (math.ceil(W/m)) + x
                print(b)
                # scatter
                for xi in range(alpha):
                    for nu in range(alpha):
                        print(V.shape, v.shape)
                        V[xi][nu][c][b] = v[xi][nu]

M = np.empty(shape=(alpha, alpha, K, P))
for xi in range(alpha):
    for nu in range(alpha):
        # perform matrix multiplication
        M[xi][nu] = U[xi][nu].dot(V[xi][nu])

# Y = np.empty(shape=(K, P, m, m))
Y = np.empty(shape=(N, K, H, W))
temp_m = np.empty(shape=(alpha, alpha))
for i in range(N):
    for k in range(K):
        # for y in range(0, H+2-m, m):
        #     for x in range(0, W+2-m, m):
        for y in range(math.ceil(H/m)):
            for x in range(math.ceil(W/m)):
                b = i * (math.ceil(H/m) * math.ceil(W/m)) + y * (math.ceil(W/m)) + x
                # gather
                for xi in range(alpha):
                    for nu in range(alpha):
                        temp_m[xi][nu] = M[xi][nu][k][b]
                # Y[k][b] =  A.T.dot(m).dot(A)
                Y[i,k,y*m:y*m+m,x*m:x*m+m] = A.T.dot(temp_m).dot(A)
            

for i in range(N):
    for k in range(K):
        # for y in range(0, H+2-m, m):
        #     for x in range(0, W+2-m, m):
        for y in range(math.ceil(H/m)):
            for x in range(math.ceil(W/m)):
                b = i * (math.ceil(H/m) * math.ceil(W/m)) + y * (math.ceil(W/m)) + x
                print(k, b)
                print(Y[i,k,y*m:y*m+m,x*m:x*m+m])

