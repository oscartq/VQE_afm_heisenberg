
import numpy as np

def get_hopping_matrix(x, y=1, tunneling=-1.0, periodic=True):
    hopping_matrix = tunneling * np.array([
        [(1.0 if (abs(i-j)%x==1) or (abs(i-j)%x==x-1) else 0.0) for i in range(x)] for j in range(x)
    ])

    return hopping_matrix

def main():
    hopping_matrix = get_hopping_matrix(8)
    print(hopping_matrix)


if __name__=='__main__':
    main()

#     hopping_matrix = -1.0 * np.array([
#             [ 0., 1., 0., 0., 0., 0., 0., 1.],
#             [ 1., 0., 1., 0., 0., 0., 0., 0.], 
#             [ 0., 1., 0., 1., 0., 0., 0., 0.], 
#             [ 0., 0., 1., 0., 1., 0., 0., 0.], 
#             [ 0., 0., 0., 1., 0., 1., 0., 0.], 
#             [ 0., 0., 0., 0., 1., 0., 1., 0.], 
#             [ 0., 0., 0., 0., 0., 1., 0., 1.], 
#             [ 1., 0., 0., 0., 0., 0., 1., 0.], 
#             ])
# elif(length==6):
#     hopping_matrix = -1.0 * np.array([
#             [ 0., 1., 0., 0., 0., 1.],
#             [ 1., 0., 1., 0., 0., 0.], 
#             [ 0., 1., 0., 1., 0., 0.], 
#             [ 0., 0., 1., 0., 1., 0.], 
#             [ 0., 0., 0., 1., 0., 1.], 
#             [ 1., 0., 0., 0., 1., 0.], 
#             ])
# else:
#     hopping_matrix = -1.0 * np.array([
#             [ 0., 1., 0., 1.],
#             [ 1., 0., 1., 0.], 
#             [ 0., 1., 0., 1.], 
#             [ 1., 0., 1., 0.], 
#             ])